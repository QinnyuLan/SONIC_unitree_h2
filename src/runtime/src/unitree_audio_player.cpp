// PCM player for the Unitree SDK2 voice service. H2 firmware compatibility
// must be checked on the target; the public SDK names this client "g1".
#include <algorithm>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/g1/audio/g1_audio_client.hpp>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <vector>

// PCM WAV parsing is independent of DDS; --check-only never connects a device.
static std::vector<uint8_t> ReadPcm(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("Cannot open WAV: " + path);
  const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(stream), {}};
  const auto u16 = [&](size_t p) -> uint16_t {
    return uint16_t(bytes.at(p)) | (uint16_t(bytes.at(p + 1)) << 8);
  };
  const auto u32 = [&](size_t p) -> uint32_t {
    return uint32_t(u16(p)) | (uint32_t(u16(p + 2)) << 16);
  };
  const auto tag = [&](size_t p, const char* value) {
    return p + 4 <= bytes.size() && std::memcmp(bytes.data() + p, value, 4) == 0;
  };
  if (bytes.size() < 12 || !tag(0, "RIFF") || !tag(8, "WAVE") ||
      uint64_t(u32(4)) + 8 != bytes.size())
    throw std::runtime_error("Invalid or truncated RIFF/WAVE file");
  bool format = false;
  std::vector<uint8_t> pcm;
  for (size_t offset = 12; offset + 8 <= bytes.size();) {
    const size_t count = u32(offset + 4), start = offset + 8;
    if (count > bytes.size() - start) throw std::runtime_error("Truncated WAV chunk");
    if (tag(offset, "fmt ")) {
      if (count < 16 || u16(start) != 1 || u16(start + 2) != 1 ||
          u32(start + 4) != 16000 || u32(start + 8) != 32000 ||
          u16(start + 12) != 2 || u16(start + 14) != 16)
        throw std::runtime_error("Require PCM16 little-endian 16 kHz mono WAV");
      format = true;
    } else if (tag(offset, "data")) {
      if (!pcm.empty()) throw std::runtime_error("Multiple WAV data chunks");
      pcm.assign(bytes.begin() + start, bytes.begin() + start + count);
    }
    offset = start + count + (count % 2);
  }
  if (!format || pcm.empty() || pcm.size() % 2) throw std::runtime_error("Invalid PCM payload");
  return pcm;
}

static volatile std::sig_atomic_t interrupted = 0;
static void on_signal(int) { interrupted = 1; }

static int Run(int argc, char** argv) {
  std::string interface, wav_path;
  int domain = 0;
  bool play = false, probe = false, check_only = false, wait_for_go = false;
  for (int i=1; i<argc; ++i) {
    const std::string option = argv[i];
    if (option == "--interface" && i+1<argc) interface = argv[++i];
    else if (option == "--domain" && i+1<argc) domain = std::stoi(argv[++i]);
    else if (option == "--play") play = true;
    else if (option == "--probe") probe = true;
    else if (option == "--wait-for-go") wait_for_go = true;
    else if (option == "--check-only") check_only = true;
    else if (!option.empty() && option[0] != '-' && wav_path.empty()) wav_path = option;
    else {
      std::cerr << "Usage: unitree_audio_player [--check-only | --probe | --play] "
                   "[--interface NIC] [--domain N] [--wait-for-go] speech.wav\n";
      return 2;
    }
  }
  if (int(play) + int(probe) + int(check_only) > 1)
    throw std::runtime_error("Choose one of --play, --probe or --check-only");
  if (wait_for_go && !play) throw std::runtime_error("--wait-for-go requires --play");
  const auto pcm = ReadPcm(wav_path);
  std::cerr << "PCM bytes=" << pcm.size() << " seconds=" << pcm.size()/32000.0 << '\n';
  if (!play && !probe) return 0;  // File validation does not initialize DDS.
  if (interface.empty()) { std::cerr << "Specify --interface explicitly\n"; return 2; }
  try {
    unitree::robot::ChannelFactory::Instance()->Init(domain, interface);
    unitree::robot::g1::AudioClient client;
    client.Init();
    client.SetTimeout(3.0f);
    uint8_t volume = 0;
    const int code = client.GetVolume(volume);
    if (code != 0) {
      std::cerr << "Unitree voice service probe failed: " << code
                << "; check H2 firmware, interface and DDS domain\n";
      return 1;
    }
    std::cerr << "Unitree voice service reachable, volume=" << int(volume) << '\n';
    if (!play) return 0;
    if (wait_for_go) {
      std::cout << "READY" << std::endl;
      std::string command;
      if (!std::getline(std::cin, command) || command != "PLAY") return 1;
    }
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    const std::string app = "sonic_replay_" + std::to_string(getpid());
    const std::string stream = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto start = std::chrono::steady_clock::now();
    const auto wait_until = [&](double seconds) {
      const auto deadline = start + std::chrono::duration<double>(seconds);
      while (!interrupted && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    };
    for (size_t offset=0; offset<pcm.size() && !interrupted; offset+=16000) {
      wait_until(std::max(0.0, offset/32000.0-0.3));
      if (interrupted) break;
      const size_t end = std::min(offset+16000, pcm.size());
      const int result = client.PlayStream(app, stream, {pcm.begin()+offset, pcm.begin()+end});
      if (result != 0) {
        client.PlayStop(app);
        std::cerr << "PlayStream failed: " << result << '\n';
        return 1;
      }
    }
    wait_until(pcm.size()/32000.0+0.5);  // Let the final queued PCM drain.
    client.PlayStop(app);             // Stop by app_name, not stream_id.
    return interrupted ? 130 : 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

int main(int argc, char** argv) {
  try { return Run(argc, argv); }
  catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
