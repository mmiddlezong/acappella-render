// Acappella Render: a VST3 effect for MuseScore's master bus that plays the
// last render made from MuseScore, following MuseScore's playback position.
//
// MuseScore tells master-bus effects where playback is (projectTimeSamples,
// the end of the current block) and whether it is playing. While the position
// moves, this plays the matching part of the render instead of MuseScore's own
// sound; when playback is stopped (or during a count-in) MuseScore's audio is
// passed through, so clicking notes still sounds.
//
// What to play comes from a small text file the MuseScore plugin writes:
//   ~/Library/Application Support/AcappellaRender/current.txt
//     audio <path to a .wav>
//     offset <seconds into the wav at MuseScore time 0>
//     render_gain <linear gain for the render>
//     musescore_gain <linear gain for MuseScore's own sound while playing>
// It is polled from a background thread and a new render is swapped in
// without stopping playback.

#include "pluginterfaces/base/ibstream.h"
#include "pluginterfaces/vst/ivstprocesscontext.h"
#include "pluginterfaces/vst/vsttypes.h"
#include "public.sdk/source/main/pluginfactory.h"
#include "public.sdk/source/vst/vstsinglecomponenteffect.h"

#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace Steinberg;
using namespace Steinberg::Vst;

namespace acappella {

static const FUID kPlayerUID(0x540D488C, 0xB5CE40AE, 0xAE9217CD, 0x2612E05F);

struct Render {
    std::vector<float> samples;   // interleaved stereo
    int64_t frames = 0;
    double rate = 44100.0;
    double offset = 0.0;          // seconds into the file at MuseScore time 0
};

static std::string dataDir()
{
    const char* home = std::getenv("HOME");
    return std::string(home ? home : "") + "/Library/Application Support/AcappellaRender";
}

static uint32_t le32(const unsigned char* p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t le16(const unsigned char* p) { return (uint16_t)(p[0] | p[1] << 8); }

// PCM 16/24/32-bit or float 32/64-bit WAV, any channel count (first two used).
static bool readWav(const std::string& path, Render& out, std::string& err)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = "can't open " + path;
        return false;
    }
    std::vector<unsigned char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (buf.size() < 12 || std::memcmp(buf.data(), "RIFF", 4) != 0 || std::memcmp(buf.data() + 8, "WAVE", 4) != 0) {
        err = "not a WAV file: " + path;
        return false;
    }
    int format = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    const unsigned char* data = nullptr;
    size_t dataLen = 0;
    size_t p = 12;
    while (p + 8 <= buf.size()) {
        const unsigned char* c = buf.data() + p;
        size_t len = le32(c + 4);
        size_t body = p + 8;
        size_t avail = std::min(len, buf.size() - body);
        if (std::memcmp(c, "fmt ", 4) == 0 && avail >= 16) {
            format = le16(c + 8);
            channels = le16(c + 10);
            rate = le32(c + 12);
            bits = le16(c + 22);
            if (format == 0xFFFE && avail >= 26) {
                format = le16(c + 8 + 24);   // first two bytes of the sub-format GUID
            }
        } else if (std::memcmp(c, "data", 4) == 0) {
            data = buf.data() + body;
            dataLen = avail;
        }
        p = body + len + (len & 1);
    }
    if (!data || channels < 1 || rate == 0) {
        err = "WAV has no audio: " + path;
        return false;
    }
    int bytes = bits / 8;
    bool ok = (format == 1 && (bits == 16 || bits == 24 || bits == 32)) || (format == 3 && (bits == 32 || bits == 64));
    if (!ok) {
        err = "unsupported WAV format " + std::to_string(format) + "/" + std::to_string(bits) + " bit";
        return false;
    }
    int64_t frames = (int64_t)(dataLen / ((size_t)bytes * channels));
    out.samples.assign((size_t)frames * 2, 0.f);
    out.frames = frames;
    out.rate = rate;
    for (int64_t i = 0; i < frames; ++i) {
        for (int ch = 0; ch < 2; ++ch) {
            int src = std::min(ch, channels - 1);
            const unsigned char* s = data + ((size_t)i * channels + src) * bytes;
            float v = 0.f;
            if (format == 3 && bits == 32) {
                std::memcpy(&v, s, 4);
            } else if (format == 3) {
                double d;
                std::memcpy(&d, s, 8);
                v = (float)d;
            } else if (bits == 16) {
                v = (int16_t)le16(s) / 32768.f;
            } else if (bits == 24) {
                int32_t x = (int32_t)(s[0] << 8 | s[1] << 16 | (uint32_t)s[2] << 24) >> 8;
                v = x / 8388608.f;
            } else {
                v = (int32_t)le32(s) / 2147483648.f;
            }
            out.samples[(size_t)i * 2 + ch] = std::isfinite(v) ? v : 0.f;
        }
    }
    return true;
}

class Player : public SingleComponentEffect
{
public:
    static FUnknown* createInstance(void*) { return (IAudioProcessor*)new Player; }

    tresult PLUGIN_API initialize(FUnknown* context) SMTG_OVERRIDE
    {
        tresult res = SingleComponentEffect::initialize(context);
        if (res != kResultOk) {
            return res;
        }
        addAudioInput(STR16("Stereo In"), SpeakerArr::kStereo);
        addAudioOutput(STR16("Stereo Out"), SpeakerArr::kStereo);
        m_stop = false;
        m_loader = std::thread([this] { loaderLoop(); });
        return kResultOk;
    }

    tresult PLUGIN_API terminate() SMTG_OVERRIDE
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_wake.notify_all();
        if (m_loader.joinable()) {
            m_loader.join();
        }
        for (Render* r : { m_incoming.exchange(nullptr), m_retired.exchange(nullptr) }) {
            delete r;
        }
        if (m_tail.r && m_tail.r != m_main.r) {
            delete m_tail.r;
        }
        delete m_main.r;
        m_main = {};
        m_tail = {};
        return SingleComponentEffect::terminate();
    }

    tresult PLUGIN_API setBusArrangements(SpeakerArrangement* inputs, int32 numIns,
                                          SpeakerArrangement* outputs, int32 numOuts) SMTG_OVERRIDE
    {
        if (numIns == 1 && numOuts == 1 && inputs[0] == SpeakerArr::kStereo && outputs[0] == SpeakerArr::kStereo) {
            return SingleComponentEffect::setBusArrangements(inputs, numIns, outputs, numOuts);
        }
        return kResultFalse;
    }

    tresult PLUGIN_API canProcessSampleSize(int32 size) SMTG_OVERRIDE
    {
        return size == kSample32 ? kResultTrue : kResultFalse;
    }

    tresult PLUGIN_API setupProcessing(ProcessSetup& setup) SMTG_OVERRIDE
    {
        tresult res = SingleComponentEffect::setupProcessing(setup);
        m_fadeStep = 1.f / std::max(1.f, (float)(0.004 * setup.sampleRate));
        m_xfadeLen = std::max(1, (int)(0.010 * setup.sampleRate));
        return res;
    }

    tresult PLUGIN_API setProcessing(TBool) SMTG_OVERRIDE { return kResultOk; }
    tresult PLUGIN_API setState(IBStream*) SMTG_OVERRIDE { return kResultOk; }
    tresult PLUGIN_API getState(IBStream*) SMTG_OVERRIDE { return kResultOk; }
    uint32 PLUGIN_API getTailSamples() SMTG_OVERRIDE { return kInfiniteTail; }
    uint32 PLUGIN_API getProcessContextRequirements() SMTG_OVERRIDE
    {
        return IProcessContextRequirements::kNeedTransportState;
    }

    tresult PLUGIN_API process(ProcessData& data) SMTG_OVERRIDE
    {
        const int32 n = data.numSamples;
        if (data.numOutputs < 1 || n <= 0 || data.outputs[0].numChannels < 1) {
            return kResultOk;
        }
        const double hostRate = processSetup.sampleRate > 0 ? processSetup.sampleRate : 44100.0;

        if (Render* fresh = m_incoming.exchange(nullptr)) {
            if (m_main.r && m_fade > 0.f) {
                startCrossfade();          // fade from the old render into the new one
            } else {
                dropTail();
                retire(m_main.r);
            }
            m_main.r = fresh;
            m_needResync = true;
        }

        const ProcessContext* ctx = data.processContext;
        const bool playing = ctx && (ctx->state & ProcessContext::kPlaying);
        const int64 pos = ctx ? ctx->projectTimeSamples : 0;
        // the position only moves once playback really runs (it stands still
        // during a count-in), and it is the END of this block
        const bool advancing = playing && pos != m_lastPos && m_main.r;
        m_lastPos = pos;

        if (advancing) {
            const int64 blockStart = std::max<int64>(0, pos - n);
            const double target = (blockStart / hostRate + m_main.r->offset) * m_main.r->rate;
            const bool jumped = std::fabs(target - m_main.pos) > 0.010 * m_main.r->rate;
            if (!m_wasAdvancing || m_needResync || jumped) {
                // MuseScore's clock rounds each block to whole microseconds,
                // so small differences are drift, not a seek: only re-sync on
                // a start, a seek/loop, a new render or real drift
                if (m_fade > 0.f && m_xfadeLeft == 0 && std::fabs(target - m_main.pos) > 1.0) {
                    startCrossfade();      // still audible: don't jump, crossfade
                }
                if (!m_wasAdvancing || jumped) {
                    m_syncHost.store(blockStart / hostRate);
                    m_syncFile.store(target / m_main.r->rate);
                    m_syncCount.fetch_add(1);
                }
                m_main.pos = target;
                m_needResync = false;
            }
        }
        m_wasAdvancing = advancing;

        AudioBusBuffers& outBus = data.outputs[0];
        float** out = outBus.channelBuffers32;
        const int nOut = outBus.numChannels;
        float** in = nullptr;
        int nIn = 0;
        if (data.numInputs > 0 && data.inputs[0].numChannels > 0) {
            in = data.inputs[0].channelBuffers32;
            nIn = data.inputs[0].numChannels;
        }
        const float renderGain = m_renderGain.load();
        const float museGain = m_museGain.load();
        const float target = advancing ? 1.f : 0.f;

        for (int32 i = 0; i < n; ++i) {
            if (m_fade < target) {
                m_fade = std::min(target, m_fade + m_fadeStep);
            } else if (m_fade > target) {
                m_fade = std::max(target, m_fade - m_fadeStep);
            }
            float l = 0.f, r = 0.f;
            if (m_main.r && m_fade > 0.f) {
                read(m_main, hostRate, l, r);
                if (m_xfadeLeft > 0 && m_tail.r) {
                    float tl, tr;
                    read(m_tail, hostRate, tl, tr);
                    const float g = (float)m_xfadeLeft / m_xfadeLen;
                    l = l * (1.f - g) + tl * g;
                    r = r * (1.f - g) + tr * g;
                    if (--m_xfadeLeft == 0) {
                        dropTail();
                    }
                }
            } else if (m_xfadeLeft > 0) {
                dropTail();
            }
            const float inL = in ? in[0][i] : 0.f;
            const float inR = in ? in[std::min(1, nIn - 1)][i] : 0.f;
            const float inGain = (1.f - m_fade) + m_fade * museGain;
            const float wet = m_fade * renderGain;
            out[0][i] = l * wet + inL * inGain;
            if (nOut > 1) {
                out[1][i] = r * wet + inR * inGain;
            }
        }
        outBus.silenceFlags = 0;
        return kResultOk;
    }

private:
    struct Voice {
        Render* r = nullptr;
        double pos = 0.0;   // in file frames
    };

    static void read(Voice& v, double hostRate, float& l, float& r)
    {
        const Render* x = v.r;
        const double p = v.pos;
        v.pos += x->rate / hostRate;
        l = r = 0.f;
        if (p < 0.0 || p >= (double)(x->frames - 1)) {
            return;
        }
        const int64_t i = (int64_t)p;
        const float f = (float)(p - (double)i);
        const float* s = x->samples.data() + i * 2;
        l = s[0] + (s[2] - s[0]) * f;
        r = s[1] + (s[3] - s[1]) * f;
    }

    void startCrossfade()
    {
        dropTail();
        m_tail = m_main;
        m_xfadeLeft = m_xfadeLen;
    }

    void dropTail()
    {
        if (m_tail.r && m_tail.r != m_main.r) {
            retire(m_tail.r);
        }
        m_tail = {};
        m_xfadeLeft = 0;
    }

    // hand a render the audio thread no longer uses to the loader thread to free
    void retire(Render* r)
    {
        if (r) {
            delete m_retired.exchange(r);   // only frees here if two pile up between polls
        }
    }

    void log(const std::string& msg)
    {
        std::time_t t = std::time(nullptr);
        std::tm tm {};
        localtime_r(&t, &tm);
        char stamp[32];
        std::strftime(stamp, sizeof stamp, "%Y-%m-%d %H:%M:%S", &tm);
        std::ofstream f(dataDir() + "/player.log", std::ios::app);
        f << stamp << "  [" << (const void*)this << "] " << msg << "\n";
    }

    void loaderLoop()
    {
        const std::string manifest = dataDir() + "/current.txt";
        ::mkdir(dataDir().c_str(), 0755);
        {
            struct stat st;
            if (::stat((dataDir() + "/player.log").c_str(), &st) == 0 && st.st_size > (1 << 20)) {
                std::remove((dataDir() + "/player.log").c_str());
            }
        }
        log("loaded");
        int64_t seenStamp = -1;
        std::string loadedKey;
        int seenSync = 0;
        std::unique_lock<std::mutex> lock(m_mutex);
        while (!m_stop) {
            lock.unlock();
            delete m_retired.exchange(nullptr);

            int sync = m_syncCount.load();
            if (sync != seenSync) {
                seenSync = sync;
                char line[128];
                std::snprintf(line, sizeof line, "play from %.3f s (render file at %.3f s)",
                              m_syncHost.load(), m_syncFile.load());
                log(line);
            }

            struct stat st;
            if (::stat(manifest.c_str(), &st) == 0) {
                int64_t stamp = (int64_t)st.st_mtimespec.tv_sec * 1000000000 + st.st_mtimespec.tv_nsec + st.st_size;
                if (stamp != seenStamp) {
                    seenStamp = stamp;
                    readManifest(manifest, loadedKey);
                }
            }

            lock.lock();
            m_wake.wait_for(lock, std::chrono::milliseconds(250), [this] { return m_stop; });
        }
    }

    void readManifest(const std::string& manifest, std::string& loadedKey)
    {
        std::ifstream f(manifest);
        std::string line, audio;
        double offset = 0.0, renderGain = 1.0, museGain = 0.0;
        while (std::getline(f, line)) {
            size_t sp = line.find(' ');
            if (sp == std::string::npos) {
                continue;
            }
            std::string key = line.substr(0, sp), val = line.substr(sp + 1);
            if (key == "audio") {
                audio = val;
            } else if (key == "offset") {
                offset = std::atof(val.c_str());
            } else if (key == "render_gain") {
                renderGain = std::atof(val.c_str());
            } else if (key == "musescore_gain") {
                museGain = std::atof(val.c_str());
            }
        }
        if (audio.empty()) {
            return;     // being rewritten; try again on the next poll
        }
        m_renderGain.store((float)std::clamp(renderGain, 0.0, 4.0));
        m_museGain.store((float)std::clamp(museGain, 0.0, 4.0));

        struct stat st;
        if (::stat(audio.c_str(), &st) != 0) {
            return;
        }
        std::ostringstream key;
        key << audio << "|" << st.st_mtimespec.tv_sec << "|" << st.st_size << "|" << offset;
        if (key.str() == loadedKey) {
            return;
        }
        Render* r = new Render;
        std::string err;
        if (!readWav(audio, *r, err)) {
            delete r;
            log("error: " + err);
            return;
        }
        r->offset = offset;
        loadedKey = key.str();
        delete m_incoming.exchange(r);
        char msg[64];
        std::snprintf(msg, sizeof msg, " (%.1f s, %.0f Hz)", r->frames / r->rate, r->rate);
        log("render " + audio + msg);
    }

    std::thread m_loader;
    std::mutex m_mutex;
    std::condition_variable m_wake;
    bool m_stop = false;

    std::atomic<Render*> m_incoming { nullptr };
    std::atomic<Render*> m_retired { nullptr };
    std::atomic<float> m_renderGain { 1.f };
    std::atomic<float> m_museGain { 0.f };
    std::atomic<double> m_syncHost { 0.0 };
    std::atomic<double> m_syncFile { 0.0 };
    std::atomic<int> m_syncCount { 0 };

    // audio thread only
    Voice m_main, m_tail;
    int m_xfadeLeft = 0;
    int m_xfadeLen = 441;
    float m_fade = 0.f;
    float m_fadeStep = 1.f / 176.f;
    int64 m_lastPos = INT64_MIN;
    bool m_wasAdvancing = false;
    bool m_needResync = false;
};

}   // namespace acappella

BEGIN_FACTORY_DEF("acappella-render", "https://github.com/mmiddlezong/acappella-render", "")

DEF_CLASS2(INLINE_UID_FROM_FUID(acappella::kPlayerUID), PClassInfo::kManyInstances, kVstAudioEffectClass,
           "Acappella Render", 0, "Fx|Generator", "1.0.0", kVstVersionString, acappella::Player::createInstance)

END_FACTORY
