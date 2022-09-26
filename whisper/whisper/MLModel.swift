//
//  MLModel.swift
//  whisper
//
//  Created by rei8 on 2022/09/24.
//

import Foundation
import AVFoundation
import Accelerate
import Combine
import SwiftUI
import CoreML

class WhisperModel: NSObject, ObservableObject {
    static let shared = WhisperModel()
        
    private override init() {
        super.init()
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            // The user has previously granted access to the camera.
            break
            
        case .notDetermined:
            /*
             The user has not yet been presented with the option to grant
             video access. Suspend the session queue to delay session
             setup until the access request has completed.
             
             Note that audio access will be implicitly requested when we
             create an AVCaptureDeviceInput for audio during session setup.
             */
            AVCaptureDevice.requestAccess(for: .audio, completionHandler: { granted in
                if !granted {
                    self.setupResult = .notAuthorized
                }
            })
            
        default:
            // The user has previously denied access.
            setupResult = .notAuthorized
        }
        try? session.setCategory(.record)
        Task.detached(priority: .userInitiated) { [self] in
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine
            do {
                encderModel = try await encoder.load(configuration: config)
                decoderModel = try await decoder.load(configuration: config)
                DispatchQueue.main.async { [self] in
                    modelLoaded = true
                }
            }
            catch {
                print(error)
            }
        }
    }
    
    var encderModel: encoder?
    var decoderModel: decoder?
    @Published var modelLoaded = false
    
    private enum SessionSetupResult {
        case success
        case notAuthorized
        case configurationFailed
    }
    
    private let engine = AVAudioEngine()
    private let session = AVAudioSession.sharedInstance()
    private let bufferQueue = DispatchQueue(label: "buffer")
    private var setupResult: SessionSetupResult = .success
    private var isSessionRunning = false
    
    private static let sample_rate = 16000
    private static let chunk_length = 30
    private static let n_fft = 512
    private static let n_mel = 80
    private static let log2n = vDSP_Length(9)
    private let fft = vDSP.FFT(log2n: log2n, radix: .radix2, ofType: DSPSplitComplex.self)
    private let window = vDSP.window(ofType: Float.self,
                                     usingSequence: .hanningDenormalized,
                                     count: n_fft,
                                     isHalfWindow: false)
    private static let hop_length = 160
    private static let n_samples = sample_rate * chunk_length
    private static let n_frames = n_samples / hop_length
    private static let filters = mel_filterbank(sr: sample_rate, n_fft: n_fft, n_mels: n_mel)
    let tokenizer = Tokenizer()
    
    private var buffer = [[Float]]()
    private var buffer_proc = 0
    private var buffer_idx = 0
    private var buffer_pad = 0
    
    private var spec_buffer = [[Float]](repeating: [Float](repeating: -1, count: n_mel), count: n_frames)
    var current_time = 0.0
    var last_plot_time = -1.0
    var prob_lang = 0.0
    var detect_lang = ""
    var prob_nospeech = 0.0
    var detect_string = ""
    var set_lang = ""
    var transcribe = true
    var loop_count = 0
    
    struct PixelData {
        var a: UInt8 = 0
        var r: UInt8 = 0
        var g: UInt8 = 0
        var b: UInt8 = 0
    }
    
    var plot: UIImage {
        let width = WhisperModel.n_frames
        let height = WhisperModel.n_mel
        
        let pixelDataSize = MemoryLayout<PixelData>.size
        var pixels = [PixelData](repeating: PixelData(), count: width*height)
        let buf = bufferQueue.sync {
            spec_buffer
        }
        for (x, spec) in buf.enumerated() {
            for (y, value) in spec.enumerated() {
                let xi = x
                let yi = height - y - 1
                let v1 = (value / 2 + 0.5) * 255
                let v = v1 > 255 ? 255 : v1 < 0 ? 0 : UInt8(v1)
                pixels[yi * width + xi] = PixelData(a:255,r:v,g:v,b:v)
            }
        }
        let data: Data = pixels.withUnsafeBufferPointer {
            return Data(buffer: $0)
        }
        let cfdata = NSData(data: data) as CFData
        guard let provider = CGDataProvider(data: cfdata) else {
            return UIImage()
        }
        guard let cgimage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * pixelDataSize,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        ) else {
            return UIImage()
        }
        return UIImage(cgImage: cgimage)
    }
    
    @ViewBuilder
    func view() -> some View {
        if !modelLoaded {
            Text("Loading...")
        }
        if isSessionRunning {
            Text("Now recording")
                .foregroundColor(.red)
        }
        Group {
            Spacer()
            Text("\(loop_count)")
            Text("t=\(current_time)")
            
            Spacer()
            Text("\(detect_lang) p=\(prob_lang)")
            Text("no speech p=\(prob_nospeech)")
            Spacer()
            Text("\(detect_string)")
                .lineLimit(nil)
                .textSelection(.enabled)
        }
    }
    
    func start() {
        if !modelLoaded {
            return
        }
        if isSessionRunning {
            return
        }
        buffer = [[Float]]()
        buffer_proc = 0
        buffer_idx = 0
        buffer_pad = 0
        current_time = 0
        last_plot_time = -1
        spec_buffer = [[Float]](repeating: [Float](repeating: -1, count: WhisperModel.n_mel), count: WhisperModel.n_frames)
        prob_lang = 0.0
        detect_lang = ""
        prob_nospeech = 0.0
        detect_string = ""

        switch setupResult {
        case .success:
            let inputNode = engine.inputNode
            let recordingFormat = inputNode.outputFormat(forBus: 0)
            guard let downFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000.0, channels: 1, interleaved: true) else {
                break
            }
            guard let converter = AVAudioConverter(from: recordingFormat, to: downFormat) else {
                break
            }
            guard let newbuffer = AVAudioPCMBuffer(pcmFormat: downFormat, frameCapacity: AVAudioFrameCount(downFormat.sampleRate * 0.1)) else {
                break
            }
            
            inputNode.installTap(onBus: 0, bufferSize: 4096, format: recordingFormat) { [weak self] (buffer, time) in
                var done = false
                let inputBlock : AVAudioConverterInputBlock = { (inNumPackets, outStatus) -> AVAudioBuffer? in
                    if done {
                        outStatus.pointee = .noDataNow
                        return nil
                    }
                    outStatus.pointee = .haveData
                    done = true
                    return buffer
                }
                var error : NSError?
                if converter.convert(to: newbuffer, error: &error, withInputFrom: inputBlock) == .haveData {
                    self?.fill_buffer(newData: newbuffer)
                }
            }
            do {
                try session.setActive(true)
                try engine.start()
                isSessionRunning = true
            }
            catch {
                print(error)
            }
        default:
            break
        }
    }
    
    func stop() {
        if !isSessionRunning {
            return
        }
        if setupResult == .success {
            engine.stop()
            engine.inputNode.removeTap(onBus: 0)
            try? session.setActive(false)
            isSessionRunning = false
            
            call_model()
        }
    }
    
    func fill_buffer(newData: AVAudioPCMBuffer) {
        let len = Int(newData.frameLength)
        let fq = newData.format.sampleRate
        guard let p = newData.floatChannelData?[0] else { return }
        let buf1 = Array(UnsafeBufferPointer(start: p, count: len))
        bufferQueue.sync {
            buffer_idx += len
            buffer.append(buf1)
            if buffer_idx > WhisperModel.sample_rate * WhisperModel.chunk_length {
                buffer_idx -= buffer[0].count
                buffer_proc -= buffer[0].count
                var drop_hop = buffer[0].count / WhisperModel.hop_length
                buffer_pad += buffer[0].count % WhisperModel.hop_length
                drop_hop += buffer_pad / WhisperModel.hop_length
                buffer_pad %= WhisperModel.hop_length
                buffer = Array(buffer.dropFirst())
                spec_buffer = spec_buffer[drop_hop...] + [[Float]](repeating: [Float](repeating: -1, count: WhisperModel.n_mel), count: drop_hop)
            }
        }
        current_time += Double(len) / fq
        DispatchQueue.global().async { [self] in
            log_mel_spectrogram()
        }
        if current_time > last_plot_time + 0.25 {
            DispatchQueue.main.async { [self] in
                objectWillChange.send()
            }
            last_plot_time = current_time
        }
    }
}

extension WhisperModel {
    static func hz_to_mel(fq: Float) -> Float {
        // Fill in the linear part
        let f_min: Float = 0.0
        let f_sp: Float = 200.0 / 3

        let mels = (fq - f_min) / f_sp

        // Fill in the log-scale part

        let min_log_hz: Float = 1000.0  // beginning of log region (Hz)
        let min_log_mel: Float = (min_log_hz - f_min) / f_sp  // same (Mels)
        let logstep: Float = log(6.4) / 27.0  // step size for log region

        if fq >= min_log_hz {
            return min_log_mel + log(fq / min_log_hz) / logstep
        }
        return mels
    }

    static func mel_to_hz(mel: Float) -> Float {
        // Fill in the linear scale
        let f_min: Float = 0.0
        let f_sp: Float = 200.0 / 3
        let freqs = f_min + f_sp * mel

        // And now the nonlinear scale
        let min_log_hz: Float = 1000.0  // beginning of log region (Hz)
        let min_log_mel: Float = (min_log_hz - f_min) / f_sp  // same (Mels)
        let logstep: Float = log(6.4) / 27.0  // step size for log region

        if mel >= min_log_mel {
            return min_log_hz * exp(logstep * (mel - min_log_mel))
        }
        return freqs
    }
    
    static func mel_filterbank(sr: Int, n_fft: Int, n_mels: Int)->[[Float]] {
        let fmin: Float = 0
        let fmax = Float(sr) / 2
        var weights = [[Float]](repeating: [Float](repeating: 0, count: 1 + n_fft / 2), count: n_mels)
        var fftfreqs = [Float](repeating: 0, count: n_fft / 2 + 1)
        for i in 0..<n_fft/2+1 {
            fftfreqs[i] = Float(i*sr) / Float(n_fft)
        }
        
        var mel_f = [Float](repeating: 0, count: n_mels + 2)
        let min_mel = hz_to_mel(fq: fmin)
        let max_mel = hz_to_mel(fq: fmax)
        for i in 0..<n_mels+2 {
            mel_f[i] = min_mel + (max_mel - min_mel) * Float(i) / Float(n_mels + 1)
        }
        mel_f = mel_f.map({ mel_to_hz(mel: $0) })
        
        var fdiff = [Float](repeating: 0, count: n_mels + 1)
        for i in 0..<n_mels+1 {
            fdiff[i] = mel_f[i+1] - mel_f[i]
        }
        
        var ramps = [Float](repeating: 0, count: mel_f.count * fftfreqs.count)
        for y in 0..<mel_f.count {
            for x in 0..<fftfreqs.count {
                let idx = y * fftfreqs.count + x
                ramps[idx] = mel_f[y] - fftfreqs[x]
            }
        }
        
        for i in 0..<n_mels {
            // lower and upper slopes for all bins
            for j in 0..<fftfreqs.count {
                let lower = -ramps[i*fftfreqs.count + j] / fdiff[i]
                let upper = ramps[(i + 2)*fftfreqs.count + j] / fdiff[i + 1]
                
                weights[i][j] = max(0, min(lower, upper))
            }
        }
        
        // Slaney-style mel is scaled to be approx constant energy per channel
        var enorm = [Float](repeating: 0, count: n_mels)
        for i in 0..<n_mels {
            enorm[i] = 2.0 / (mel_f[2 + i] - mel_f[i])
        }
        for i in 0..<n_mels {
            for j in 0..<fftfreqs.count {
                weights[i][j] *= enorm[i]
            }
        }
        return weights
    }
    
    func calc_FFT(signal: [Float]) -> [Float] {
        let signal = vDSP.multiply(signal, window)
        let count = WhisperModel.n_fft / 2
        let magnitudes = [Float](unsafeUninitializedCapacity: count+1) {
            buffer, initializedCount in

            var realParts = [Float](repeating: 0, count: count)
            var imagParts = [Float](repeating: 0, count: count)
            realParts.withUnsafeMutableBufferPointer { realPtr in
                imagParts.withUnsafeMutableBufferPointer { imagPtr in
                    var complexSignal = DSPSplitComplex(realp: realPtr.baseAddress!,
                                                        imagp: imagPtr.baseAddress!)

                    signal.withUnsafeBytes {
                        vDSP.convert(interleavedComplexVector: [DSPComplex]($0.bindMemory(to: DSPComplex.self)),
                                     toSplitComplexVector: &complexSignal)
                    }

                    fft?.forward(input: complexSignal,
                                 output: &complexSignal)
                    vDSP.squareMagnitudes(complexSignal,
                                          result: &buffer)
                }
            }
            buffer[0] = realParts[0]
            buffer[count] = imagParts[0]
            initializedCount = count + 1
        }
        return magnitudes
    }

    func get_buffer(st: Int) -> [Float] {
        var idx = st
        var ret = [Float]()
        var len = WhisperModel.n_fft
        if idx < buffer_pad {
            idx = min(-idx, len)
            ret += [Float](repeating: 0, count: idx)
            len -= idx
            idx = 0
        }
        var k = buffer_pad
        let copyBuffer = bufferQueue.sync {
            buffer
        }
        for buf in copyBuffer {
            if idx >= k && idx < k + buf.count {
                let s = idx - k
                if s + len <= buf.count {
                    ret += buf[s..<s+len]
                    return ret
                }
                ret += buf[s...]
                len -= buf.count - s
                idx += buf.count - s
            }
            k += buf.count
        }
        if len > 0 {
            ret += [Float](repeating: 0, count: len)
        }
        return ret
    }
    
    func log_mel_spectrogram() {
        let hop_start = max(buffer_proc / WhisperModel.hop_length, 0)
        let hop_end = min(buffer_idx / WhisperModel.hop_length - 1, WhisperModel.n_frames)
        if hop_end < hop_start {
            return
        }
        DispatchQueue.concurrentPerform(iterations: hop_end - hop_start) { i in
            let k = i + hop_start
            let t = k * WhisperModel.hop_length
            let st = t - WhisperModel.n_fft / 2
            let mag = calc_FFT(signal: get_buffer(st: st))
            var log_spec = (0..<WhisperModel.n_mel).map({ m in
                let v = vDSP.sum(vDSP.multiply(WhisperModel.filters[m], mag))
                return log10(max(v, 1e-10))
            })
            let min_v = vDSP.maximum(log_spec) - 8
            log_spec = log_spec.map({ (max($0, min_v) + 4) / 4 })
            bufferQueue.sync {
                spec_buffer[k] = log_spec
            }
        }
        buffer_proc = hop_end * WhisperModel.hop_length
    }
}

extension WhisperModel {
    func call_model() {
        Task.detached(priority: .userInitiated) { [self] in
            guard let encderModel = encderModel else {
                return
            }
            guard let decoderModel = decoderModel else {
                return
            }
            let spec = bufferQueue.sync {
                spec_buffer
            }
            var mel = MLShapedArray(repeating: Float(-2.0), shape: [1,80,3000])
            mel.withUnsafeMutableShapedBufferPointer { ptr, shape, stride in
                for (i,s) in spec.enumerated() {
                    for j in 0..<shape[1] {
                        ptr[j*stride[1]+i] = s[j]
                    }
                }
            }
            guard let soundoutput = try? encderModel.prediction(input: encoderInput(mel: mel)) else {
                return
            }
            let token = MLShapedArray(repeating: Int32(tokenizer.sot), shape: [1,1])
            let input_decoder = decoderInput(tokens: token, audio_features: soundoutput.audio_featuresShapedArray)
            
            guard let output = try? decoderModel.prediction(input: input_decoder) else {
                return
            }
            
            let prob = tokenizer.getProbLanguage(logits: output.logitsShapedArray)
            guard let lang = prob.sorted(by: {$0.value > $1.value}).first else {
                return
            }
            
            prob_lang = Double(lang.value)
            detect_lang = lang.key
            var tokens: [Int] = []
            
            for i in 0..<440 {
                loop_count = i
                DispatchQueue.main.async {
                    self.objectWillChange.send()
                }

                var token = MLShapedArray(repeating: Int32(0), shape: [1,3+i])
                token.withUnsafeMutableShapedBufferPointer { ptr, shape, stride in
                    ptr[0] = Int32(tokenizer.sot)
                    if set_lang != "" {
                        ptr[1] = Int32(tokenizer.getLanguageToken(lang: set_lang))
                    }
                    else {
                        ptr[1] = Int32(tokenizer.getLanguageToken(lang: detect_lang))
                    }
                    if transcribe {
                        ptr[2] = Int32(tokenizer.sot_transcribe)
                    }
                    else {
                        ptr[2] = Int32(tokenizer.sot_translate)
                    }
                    for j in 0..<i {
                        ptr[j+3] = Int32(tokens[j])
                    }
                }

                let input_decoder = decoderInput(tokens: token, audio_features: soundoutput.audio_featuresShapedArray)

                guard let output = try? decoderModel.prediction(input: input_decoder) else {
                    return
                }
                if i == 0 {
                    prob_nospeech = Double(tokenizer.getNoSpeechProb(logits: output.logitsShapedArray))
                }
                
                let t = tokenizer.getLastToken(logits: output.logitsShapedArray)
                if t == tokenizer.eot {
                    break
                }
                tokens.append(t)
            }
            
            print(tokens)
            detect_string = tokenizer.decodeTokens(tokens: tokens)
            
            DispatchQueue.main.async {
                self.objectWillChange.send()
            }
        }
    }
}
