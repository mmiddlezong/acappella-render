// Acappella Render: VST3 effects that play the last render made from MuseScore,
// following MuseScore's playback position. This file builds two of them:
//
// "Acappella Render", for MuseScore's master bus. MuseScore tells master-bus
// effects where playback is (projectTimeSamples, the end of the current
// block) and whether it is playing. While the position moves, this plays the
// matching part of the render instead of MuseScore's own sound; when playback
// is stopped (or during a count-in) MuseScore's audio is passed through, so
// clicking notes still sounds.
//
// "Acappella Render Part" (built with ACAPPELLA_PART), for a part's own mixer
// channel. It is silent while the render plays (MuseScore's sound otherwise),
// and MuseScore stops calling a channel's effects once the channel is muted
// (or another part soloed) and silent. So while a part's effect isn't being
// called, the master-bus effect leaves that part's own track out of what it
// plays. The two find each other through a small table in shared memory.
// Each part effect has an id, kept in its state, which MuseScore saves with
// the score; the renderer reads the ids from there to say which part's track
// each one controls.
//
// What to play comes from a small text file the MuseScore plugin writes:
//   ~/Library/Application Support/AcappellaRender/current.txt
//     audio <path to a .wav>       the whole render
//     offset <seconds into the wav at MuseScore time 0>
//     render_gain <linear gain for the render>
//     musescore_gain <linear gain for MuseScore's own sound while playing>
//   and, when the score has part effects:
//     rate <sample rate of the part tracks>
//     stem_gain <gain the part tracks are played at>
//     stem <ids or -> <path to a .npy>   one part's track (float32 stereo
//                                        frames) and the part effects (ids,
//                                        comma-separated) that control it
// It is polled from a background thread and a new render is swapped in
// without stopping playback. With part tracks, the whole render is their sum.

#include "pluginterfaces/base/ibstream.h"
#include "pluginterfaces/vst/ivstprocesscontext.h"
#include "pluginterfaces/vst/vsttypes.h"
#include "public.sdk/source/main/pluginfactory.h"
#include "public.sdk/source/vst/vstsinglecomponenteffect.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
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
static const FUID kPartUID(0x82CAF363, 0x83E24A39, 0xB0A1CC58, 0xDED6308C);

static std::string dataDir()
{
    const char* home = std::getenv("HOME");
    return std::string(home ? home : "") + "/Library/Application Support/AcappellaRender";
}

static void log(const void* who, const std::string& msg)
{
    std::time_t t = std::time(nullptr);
    std::tm tm {};
    localtime_r(&t, &tm);
    char stamp[32];
    std::strftime(stamp, sizeof stamp, "%Y-%m-%d %H:%M:%S", &tm);
    std::ofstream f(dataDir() + "/player.log", std::ios::app);
    f << stamp << "  [" << who << "] " << msg << "\n";
}

static int64_t nowNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- registry
// Every effect of either kind claims a slot in a table in shared memory (the
// two kinds are separate bundles, so they share no memory otherwise). Only
// slots of this process count; a dead process's slots are taken over.

enum : uint32_t { kRoleMaster = 1, kRolePart = 2 };
enum : uint32_t { kHasRender = 1, kPlayingRender = 2 };

struct Slot {
    std::atomic<uint64_t> owner;    // pid << 32 | serial; 0 = free
    std::atomic<int64_t> seen;      // when process() last ran (steady clock, ns)
    std::atomic<int64_t> cycle;     // master: blocks processed; part: the master's
                                    // count when this part last ran
    std::atomic<uint32_t> role;
    std::atomic<uint32_t> flags;    // master: kHasRender, kPlayingRender
    std::atomic<float> museGain;    // master: MuseScore's sound under the render
    char id[33];                    // part: its id (32 hex digits)
};

static constexpr int kSlots = 128;
struct Registry {
    Slot slots[kSlots];
};

static Registry* registry()
{
    static Registry* reg = []() -> Registry* {
        char name[32];
        std::snprintf(name, sizeof name, "/acrender1-%u", (unsigned)getuid());
        int fd = shm_open(name, O_CREAT | O_RDWR, 0600);
        if (fd < 0) {
            return nullptr;
        }
        struct stat st;
        bool ok = fstat(fd, &st) == 0
                  && (st.st_size >= (off_t)sizeof(Registry) || ftruncate(fd, sizeof(Registry)) == 0);
        void* p = ok ? mmap(nullptr, sizeof(Registry), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0) : MAP_FAILED;
        close(fd);
        return p == MAP_FAILED ? nullptr : static_cast<Registry*>(p);
    }();
    return reg;
}

static uint32_t ownerPid(uint64_t owner) { return (uint32_t)(owner >> 32); }

static bool mine(const Slot& s)
{
    return ownerPid(s.owner.load(std::memory_order_acquire)) == (uint32_t)getpid();
}

static Slot* claimSlot(uint32_t role, const char* id)
{
    Registry* reg = registry();
    if (!reg) {
        return nullptr;
    }
    static std::atomic<uint32_t> serial { 0 };
    const uint64_t me = (uint64_t)getpid() << 32 | (serial.fetch_add(1) + 1);
    for (Slot& s : reg->slots) {
        uint64_t cur = s.owner.load();
        bool free = cur == 0;
        if (!free && ownerPid(cur) != (uint32_t)getpid()) {
            free = kill((pid_t)ownerPid(cur), 0) != 0 && errno == ESRCH;   // left by a dead process
        }
        if (free && s.owner.compare_exchange_strong(cur, me)) {
            std::memset(s.id, 0, sizeof s.id);
            if (id) {
                std::strncpy(s.id, id, 32);
            }
            s.flags.store(0);
            s.cycle.store(-1000);
            s.museGain.store(0.f);
            s.seen.store(nowNs());
            s.role.store(role, std::memory_order_release);
            return &s;
        }
    }
    return nullptr;
}

static void releaseSlot(Slot* s)
{
    if (s) {
        s->role.store(0);
        s->owner.store(0, std::memory_order_release);
    }
}

// a part effect with this id: does it exist in this process, and did it
// run in the last few of the master's blocks (MuseScore runs a block's part
// effects before the master bus's)?
static void partState(const std::string& id, int64_t cycle, bool& present, bool& alive)
{
    Registry* reg = registry();
    if (!reg) {
        return;
    }
    for (const Slot& s : reg->slots) {
        if (s.role.load(std::memory_order_acquire) != kRolePart || !mine(s)
            || std::strncmp(s.id, id.c_str(), 32) != 0) {
            continue;
        }
        present = true;
        if (s.cycle.load(std::memory_order_relaxed) >= cycle - 3) {
            alive = true;
        }
    }
}

// the master-bus effect of this process, if it's running: its flags, its
// block count and how much of MuseScore's sound it keeps under the render
static uint32_t masterState(int64_t now, float& museGain, int64_t& cycle)
{
    Registry* reg = registry();
    if (!reg) {
        return 0;
    }
    for (const Slot& s : reg->slots) {
        if (s.role.load(std::memory_order_acquire) != kRoleMaster || !mine(s)
            || now - s.seen.load(std::memory_order_relaxed) > 250000000) {
            continue;
        }
        museGain = s.museGain.load(std::memory_order_relaxed);
        cycle = s.cycle.load(std::memory_order_relaxed);
        return s.flags.load(std::memory_order_relaxed);
    }
    return 0;
}

// ------------------------------------------------------------------ renders
struct Stem {
    std::vector<int16_t> samples;    // interleaved stereo, times scale
    float scale = 0.f;
    int64_t count = 0;
    std::vector<std::string> ids;    // the part effects that control it
    float gain = 1.f;                // audio thread: now, and where it's going
    float target = 1.f;
};

struct Render {
    std::vector<float> samples;   // interleaved stereo (without part tracks)
    std::vector<Stem> stems;      // or each part's track
    float stemGain = 1.f;
    int64_t frames = 0;
    double rate = 44100.0;
    double offset = 0.0;          // seconds into the file at MuseScore time 0

};

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

// A part's track: a .npy file of float32 frames, shape (n, 2). It's read in
// one go (through a memory map, playback could end up waiting for the disk)
// and kept as 16-bit samples scaled to the track's own peak.
static bool readNpy(const std::string& path, Stem& out, std::string& err)
{
    std::ifstream f(path, std::ios::binary);
    unsigned char pre[12] = {};
    if (!f || !f.read((char*)pre, sizeof pre) || std::memcmp(pre, "\x93NUMPY", 6) != 0) {
        err = "not a track: " + path;
        return false;
    }
    const size_t start = pre[6] == 1 ? 10 : 12;
    const size_t headerLen = pre[6] == 1 ? le16(pre + 8) : le32(pre + 8);
    std::string header(headerLen, '\0');
    f.seekg((std::streamoff)start);
    f.read(&header[0], (std::streamsize)headerLen);
    const size_t shape = header.find("'shape': (");
    long long n = 0, width = 0;
    if (!f || header.find("'descr': '<f4'") == std::string::npos
        || header.find("'fortran_order': False") == std::string::npos || shape == std::string::npos
        || std::sscanf(header.c_str() + shape + 10, "%lld, %lld", &n, &width) != 2 || width != 2 || n < 2) {
        err = "not a stereo float32 track: " + path;
        return false;
    }
    std::vector<float> raw((size_t)n * 2);
    if (!f.read(reinterpret_cast<char*>(raw.data()), (std::streamsize)(raw.size() * sizeof(float)))) {
        err = "track cut short: " + path;
        return false;
    }
    float peak = 0.f;
    for (float& v : raw) {
        if (!std::isfinite(v)) {
            v = 0.f;
        }
        peak = std::max(peak, std::fabs(v));
    }
    const float toInt = peak > 0.f ? 32767.f / peak : 0.f;
    out.samples.resize(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        out.samples[i] = (int16_t)std::lrint(raw[i] * toInt);
    }
    out.scale = peak / 32767.f;
    out.count = n;
    return true;
}

// -------------------------------------------------------------- master bus
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
        m_slot = claimSlot(kRoleMaster, nullptr);
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
        releaseSlot(m_slot);
        m_slot = nullptr;
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
        m_gainStep = 1.f / std::max(1.f, (float)(0.010 * setup.sampleRate));
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
        const int64_t now = nowNs();
        const int64_t cycle = ++m_cycle;
        if (m_slot) {
            m_slot->cycle.store(cycle, std::memory_order_relaxed);
        }

        if (Render* fresh = m_incoming.exchange(nullptr)) {
            if (m_main.r && m_fade > 0.f) {
                startCrossfade();          // fade from the old render into the new one
            } else {
                dropTail();
                retire(m_main.r);
            }
            m_main.r = fresh;
            m_needResync = true;
            updateTargets(fresh, cycle);
            for (Stem& s : fresh->stems) {
                s.gain = s.target;          // start where the parts are, no fade
            }
        }
        if (m_main.r) {
            updateTargets(m_main.r, cycle);
        }
        if (m_tail.r && m_tail.r != m_main.r) {
            updateTargets(m_tail.r, cycle);
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

        const float renderGain = m_renderGain.load();
        const float museGain = m_museGain.load();
        if (m_slot) {
            m_slot->museGain.store(museGain, std::memory_order_relaxed);
            m_slot->flags.store((m_main.r ? kHasRender : 0) | (advancing ? kPlayingRender : 0),
                                std::memory_order_relaxed);
            m_slot->seen.store(now, std::memory_order_release);
        }

        AudioBusBuffers& outBus = data.outputs[0];
        float** out = outBus.channelBuffers32;
        const int nOut = outBus.numChannels;
        float** in = nullptr;
        int nIn = 0;
        if (data.numInputs > 0 && data.inputs[0].numChannels > 0) {
            in = data.inputs[0].channelBuffers32;
            nIn = data.inputs[0].numChannels;
        }
        const float target = advancing ? 1.f : 0.f;

        for (int32 i = 0; i < n; ++i) {
            if (m_fade < target) {
                m_fade = std::min(target, m_fade + m_fadeStep);
            } else if (m_fade > target) {
                m_fade = std::max(target, m_fade - m_fadeStep);
            }
            rampGains(m_main.r);
            if (m_tail.r != m_main.r) {
                rampGains(m_tail.r);
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

    // which part tracks to play: all but those whose part effects exist and
    // aren't being called (muted parts); one bit each for the log
    void updateTargets(Render* r, int64_t cycle)
    {
        uint64_t heard = 0;
        for (size_t k = 0; k < r->stems.size(); ++k) {
            Stem& s = r->stems[k];
            bool present = false, alive = false;
            for (const std::string& id : s.ids) {
                partState(id, cycle, present, alive);
            }
            s.target = !present || alive ? 1.f : 0.f;
            if (s.target > 0.f && k < 64) {
                heard |= uint64_t(1) << k;
            }
        }
        if (r == m_main.r) {
            m_heard.store(heard, std::memory_order_relaxed);
        }
    }

    void rampGains(Render* r)
    {
        if (!r) {
            return;
        }
        for (Stem& s : r->stems) {
            if (s.gain < s.target) {
                s.gain = std::min(s.target, s.gain + m_gainStep);
            } else if (s.gain > s.target) {
                s.gain = std::max(s.target, s.gain - m_gainStep);
            }
        }
    }

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
        if (x->stems.empty()) {
            const float* s = x->samples.data() + i * 2;
            l = s[0] + (s[2] - s[0]) * f;
            r = s[1] + (s[3] - s[1]) * f;
            return;
        }
        for (const Stem& st : x->stems) {
            if (st.gain <= 0.f || i + 1 >= st.count) {
                continue;
            }
            const int16_t* s = st.samples.data() + i * 2;
            const float g = st.gain * st.scale;
            l += ((float)s[0] + (float)(s[2] - s[0]) * f) * g;
            r += ((float)s[1] + (float)(s[3] - s[1]) * f) * g;
        }
        l *= x->stemGain;
        r *= x->stemGain;
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

    void log(const std::string& msg) { acappella::log(this, msg); }

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
        uint64_t seenHeard = ~uint64_t(0);
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
            const uint64_t heard = m_heard.load();
            if (heard != seenHeard && m_stemCount > 0) {
                if (seenHeard != ~uint64_t(0)) {
                    std::string parts;
                    for (int k = 0; k < m_stemCount && k < 64; ++k) {
                        parts += (heard >> k & 1) ? " +" : " -";
                        parts += std::to_string(k + 1);
                    }
                    log("parts heard:" + parts);
                }
                seenHeard = heard;
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
        double offset = 0.0, renderGain = 1.0, museGain = 0.0, rate = 44100.0, stemGain = 1.0;
        std::vector<std::pair<std::string, std::string>> stems;   // (ids, path)
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
            } else if (key == "rate") {
                rate = std::atof(val.c_str());
            } else if (key == "stem_gain") {
                stemGain = std::atof(val.c_str());
            } else if (key == "stem") {
                size_t sp2 = val.find(' ');
                if (sp2 != std::string::npos) {
                    stems.emplace_back(val.substr(0, sp2), val.substr(sp2 + 1));
                }
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
        key << audio << "|" << st.st_mtimespec.tv_sec << "|" << st.st_size << "|" << offset << "|" << stemGain;
        for (const auto& s : stems) {
            struct stat ss;
            key << "|" << s.first << "|" << s.second << "|" << (::stat(s.second.c_str(), &ss) == 0 ? ss.st_size : -1);
        }
        if (key.str() == loadedKey) {
            return;
        }
        Render* r = new Render;
        std::string err;
        bool ok = !stems.empty();
        for (const auto& s : stems) {
            Stem stem;
            if (!readNpy(s.second, stem, err)) {
                ok = false;
                break;
            }
            std::stringstream ids(s.first);
            std::string id;
            while (std::getline(ids, id, ',')) {
                if (!id.empty() && id != "-") {
                    stem.ids.push_back(id);
                }
            }
            r->frames = std::max(r->frames, stem.count);
            r->stems.push_back(std::move(stem));
        }
        if (ok) {
            r->rate = rate > 0 ? rate : 44100.0;
            r->stemGain = (float)stemGain;
        } else {
            if (!stems.empty()) {
                log("error: " + err + "; playing the whole render");
            }
            delete r;
            r = new Render;
            if (!readWav(audio, *r, err)) {
                delete r;
                log("error: " + err);
                return;
            }
        }
        r->offset = offset;
        loadedKey = key.str();
        m_stemCount = (int)r->stems.size();
        delete m_incoming.exchange(r);
        char msg[96];
        std::snprintf(msg, sizeof msg, " (%.1f s, %.0f Hz%s)", r->frames / r->rate, r->rate,
                      r->stems.empty() ? "" : (", " + std::to_string(r->stems.size()) + " part tracks").c_str());
        log("render " + audio + msg);
    }

    std::thread m_loader;
    std::mutex m_mutex;
    std::condition_variable m_wake;
    bool m_stop = false;
    int m_stemCount = 0;              // loader thread

    Slot* m_slot = nullptr;
    std::atomic<Render*> m_incoming { nullptr };
    std::atomic<Render*> m_retired { nullptr };
    std::atomic<float> m_renderGain { 1.f };
    std::atomic<float> m_museGain { 0.f };
    std::atomic<double> m_syncHost { 0.0 };
    std::atomic<double> m_syncFile { 0.0 };
    std::atomic<int> m_syncCount { 0 };
    std::atomic<uint64_t> m_heard { 0 };

    // audio thread only
    Voice m_main, m_tail;
    int m_xfadeLeft = 0;
    int m_xfadeLen = 441;
    float m_fade = 0.f;
    float m_fadeStep = 1.f / 176.f;
    float m_gainStep = 1.f / 441.f;
    int64 m_lastPos = INT64_MIN;
    int64_t m_cycle = 0;
    bool m_wasAdvancing = false;
    bool m_needResync = false;
};

// ------------------------------------------------------------- a part's channel
class PartEffect : public SingleComponentEffect
{
public:
    static FUnknown* createInstance(void*) { return (IAudioProcessor*)new PartEffect; }

    PartEffect()
    {
        unsigned char raw[16];
        arc4random_buf(raw, sizeof raw);
        for (int i = 0; i < 16; ++i) {
            std::snprintf(m_id + 2 * i, 3, "%02x", raw[i]);
        }
    }

    tresult PLUGIN_API initialize(FUnknown* context) SMTG_OVERRIDE
    {
        tresult res = SingleComponentEffect::initialize(context);
        if (res != kResultOk) {
            return res;
        }
        addAudioInput(STR16("Stereo In"), SpeakerArr::kStereo);
        addAudioOutput(STR16("Stereo Out"), SpeakerArr::kStereo);
        m_slot = claimSlot(kRolePart, m_id);
        m_stop = false;
        m_waiter = std::thread([this] { askToBeSaved(); });
        return kResultOk;
    }

    tresult PLUGIN_API terminate() SMTG_OVERRIDE
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_wake.notify_all();
        if (m_waiter.joinable()) {
            m_waiter.join();
        }
        releaseSlot(m_slot);
        m_slot = nullptr;
        return SingleComponentEffect::terminate();
    }

    tresult PLUGIN_API setBusArrangements(SpeakerArrangement* inputs, int32 numIns,
                                          SpeakerArrangement* outputs, int32 numOuts) SMTG_OVERRIDE
    {
        if (numIns == 1 && numOuts == 1 && inputs[0] == outputs[0]
            && (inputs[0] == SpeakerArr::kStereo || inputs[0] == SpeakerArr::kMono)) {
            return SingleComponentEffect::setBusArrangements(inputs, numIns, outputs, numOuts);
        }
        return kResultFalse;
    }

    tresult PLUGIN_API canProcessSampleSize(int32 size) SMTG_OVERRIDE
    {
        return size == kSample32 ? kResultTrue : kResultFalse;
    }

    tresult PLUGIN_API setProcessing(TBool) SMTG_OVERRIDE { return kResultOk; }
    uint32 PLUGIN_API getTailSamples() SMTG_OVERRIDE { return 0; }

    // the id, which MuseScore keeps with the score
    tresult PLUGIN_API getState(IBStream* state) SMTG_OVERRIDE
    {
        if (!state) {
            return kResultFalse;
        }
        char buf[36];
        std::memcpy(buf, "ACRP", 4);
        std::memcpy(buf + 4, m_id, 32);
        int32 written = 0;
        return state->write(buf, sizeof buf, &written) == kResultOk && written == (int32)sizeof buf
                   ? kResultOk : kResultFalse;
    }

    tresult PLUGIN_API setState(IBStream* state) SMTG_OVERRIDE
    {
        char buf[36];
        int32 got = 0;
        if (!state || state->read(buf, sizeof buf, &got) != kResultOk || got != (int32)sizeof buf
            || std::memcmp(buf, "ACRP", 4) != 0) {
            return kResultFalse;
        }
        std::memcpy(m_id, buf + 4, 32);
        m_id[32] = 0;
        if (m_slot) {
            std::memcpy(m_slot->id, m_id, 33);
        }
        m_restored = true;
        return kResultOk;
    }

    tresult PLUGIN_API process(ProcessData& data) SMTG_OVERRIDE
    {
        const int64_t now = nowNs();
        float museGain = 0.f;
        int64_t cycle = -1000;
        const uint32_t master = masterState(now, museGain, cycle);
        if (m_slot) {
            m_slot->cycle.store(cycle, std::memory_order_relaxed);
            m_slot->seen.store(now, std::memory_order_release);
        }
        const int32 n = data.numSamples;
        if (data.numOutputs < 1 || n <= 0) {
            return kResultOk;
        }
        // silent while the master bus plays the render (so a muted channel
        // falls silent at once and MuseScore stops calling this), unless
        // MuseScore's own sound is kept underneath it
        const bool quiet = (master & kPlayingRender) && museGain <= 0.f;
        AudioBusBuffers& outBus = data.outputs[0];
        for (int32 c = 0; c < outBus.numChannels; ++c) {
            float* o = outBus.channelBuffers32[c];
            const float* in = data.numInputs > 0 && c < data.inputs[0].numChannels
                                  ? data.inputs[0].channelBuffers32[c] : nullptr;
            if (quiet || !in) {
                std::memset(o, 0, sizeof(float) * n);
            } else if (o != in) {
                std::memcpy(o, in, sizeof(float) * n);
            }
        }
        outBus.silenceFlags = quiet ? ((uint64)1 << outBus.numChannels) - 1 : 0;
        return kResultOk;
    }

private:
    // A new effect asks MuseScore to keep its state (its id) with the score,
    // once MuseScore has had the chance to give it a saved one.
    void askToBeSaved()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_wake.wait_for(lock, std::chrono::seconds(2), [this] { return m_stop; })) {
            return;
        }
        if (!m_restored) {
            setDirty(true);
            acappella::log(this, std::string("part effect ") + m_id + " asked to be saved with the score");
        }
    }

    char m_id[33] = {};
    std::atomic<bool> m_restored { false };
    Slot* m_slot = nullptr;
    std::thread m_waiter;
    std::mutex m_mutex;
    std::condition_variable m_wake;
    bool m_stop = false;
};

}   // namespace acappella

BEGIN_FACTORY_DEF("acappella-render", "https://github.com/mmiddlezong/acappella-render", "")

#ifdef ACAPPELLA_PART
DEF_CLASS2(INLINE_UID_FROM_FUID(acappella::kPartUID), PClassInfo::kManyInstances, kVstAudioEffectClass,
           "Acappella Render Part", 0, "Fx", "1.0.0", kVstVersionString, acappella::PartEffect::createInstance)
#else
DEF_CLASS2(INLINE_UID_FROM_FUID(acappella::kPlayerUID), PClassInfo::kManyInstances, kVstAudioEffectClass,
           "Acappella Render", 0, "Fx|Generator", "1.0.0", kVstVersionString, acappella::Player::createInstance)
#endif

END_FACTORY
