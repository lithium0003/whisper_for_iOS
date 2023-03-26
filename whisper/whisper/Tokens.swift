//
//  Tokens.swift
//  whisper
//
//  Created by rei8 on 2022/09/25.
//

import Foundation
import CoreML

class Tokenizer {
    let vocab: [String: Int]
    let decoder: [String]
    let codemap: [Character: Int]

    init() {
        guard let url = Bundle.main.url(forResource: "vocab", withExtension: "json") else {
            vocab = [:]
            decoder = []
            codemap = [:]
            return
        }
        guard let data = try? Data(contentsOf: url) else {
            vocab = [:]
            decoder = []
            codemap = [:]
            return
        }
        guard let vocab = try? JSONDecoder().decode(Dictionary<String,Int>.self, from: data) else {
            self.vocab = [:]
            decoder = []
            codemap = [:]
            return
        }
        self.vocab = vocab
        var reverce_map: [Int: String] = [:]
        for (k,v) in vocab {
            reverce_map[v] = k
        }
        decoder = reverce_map.sorted(by: { $0.key < $1.key }).map({ $0.value })
        
        var bs = [("!", "~"), ("¡", "¬"), ("®", "ÿ")].map({
            let st = $0.0.unicodeScalars.first?.value ?? 0
            let ed = $0.1.unicodeScalars.first?.value ?? 0
            return (st...ed).map({ $0 })
        }).reduce([UInt32](), { $0 + $1 })
        var cs = bs
        var n: UInt32 = 0
        for b: UInt32 in 0..<256 {
            if !bs.contains(b) {
                bs.append(b)
                cs.append(256 + n)
                n += 1
            }
        }
        codemap = Dictionary(uniqueKeysWithValues: zip(cs, bs).map{ (Character( UnicodeScalar($0) ?? " "), Int($1)) })
    }

    let eot = 50257
    let sot = 50258
    let sot_translate = 50358
    let sot_transcribe = 50359
    let sot_lm = 50360
    let sot_prev = 50361
    let no_speech = 50362
    let no_timestamps = 50363
    let timestamp_begin = 50364

    let language_codes: [String] =
        ["en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv",
         "it", "id", "hi", "fi", "vi", "iw", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
         "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr",
         "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
         "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu",
         "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
         "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"]


    let language_tokens: [Int] = Array(50259..<50358)
    func getLanguageToken(lang: String) -> Int {
        language_tokens[language_codes.firstIndex(of: lang) ?? 0]
    }

    func getProbLanguage(logits: MLShapedArray<Float>, t_idx: Int = 0) -> [String: Float] {
        var expsum: Float = 0
        var logit_value: [Int: Float] = [:]
        logits.withUnsafeShapedBufferPointer { ptr, shape, stride in
            assert(t_idx < shape[1])
            for lang in language_tokens {
                logit_value[lang] = ptr[stride[1]*t_idx + lang]
            }
        }
        let max_x = logit_value.map({ $0.value }).max(by: <) ?? -Float.infinity
        for (_, value) in logit_value {
            expsum += exp(value - max_x)
        }
        return Dictionary(uniqueKeysWithValues: logit_value.map({
            (language_codes[language_tokens.firstIndex(of: $0.key)!],
             exp($0.value - max_x) / expsum
            )
        }))
    }

    func getNoSpeechProb(logits: MLShapedArray<Float>, t_idx: Int = 0) -> Float {
        logits.withUnsafeShapedBufferPointer { ptr, shape, stride in
            assert(t_idx < shape[1])
            var max_value = -Float.infinity
            for i in 0..<shape[2] {
                let value = ptr[stride[1]*t_idx + i]
                if value > max_value {
                    max_value = value
                }
            }
            var expsum: Float = 0
            for i in 0..<shape[2] {
                let value = ptr[stride[1]*t_idx + i]
                expsum += exp(value - max_value)
            }
            return exp(ptr[stride[1]*t_idx + no_speech] - max_value) / expsum
        }
    }

    func ApplyTimestampRules(tokens: [Int], logits: [(offset: Int, element: Float)]) -> [(offset: Int, element: Float)] {
        var ret = logits.filter({ $0.offset != no_timestamps })
        var last_was_timestamp = false
        var penultimate_was_timestamp = false
        if tokens.count >= 1, tokens[tokens.count - 1] >= timestamp_begin {
            last_was_timestamp = true
        }
        if tokens.count < 2 || tokens[tokens.count - 2] >= timestamp_begin {
            penultimate_was_timestamp = true
        }
        if last_was_timestamp {
            if penultimate_was_timestamp {
                ret = ret.filter({ $0.offset < timestamp_begin })
            }
            else {
                ret = ret.filter({ $0.offset >= eot })
            }
        }
        
        let timestamps = tokens.filter({ $0 >= timestamp_begin })
        if timestamps.count > 0 {
            ret = ret.filter({ $0.offset < timestamp_begin || $0.offset >= timestamps.last! })
        }
        
        if tokens.isEmpty {
            ret = ret.filter({ $0.offset >= timestamp_begin })
        }
        
        var expsum: Float = 0
        let max_x = ret.max(by: { $0.element < $1.element })?.element ?? -Float.infinity
        for item in ret {
            expsum += exp(item.element - max_x)
        }
        let logprobs = log_softmax(ret)
        let timestamp_logprob = logsumexp(logprobs.filter({ $0.offset >= timestamp_begin }).map({ $0.element }))
        let max_text_token_logprob = logprobs.filter({ $0.offset < timestamp_begin }).map({ $0.element }).max() ?? -Float.infinity

        if timestamp_logprob > max_text_token_logprob {
            ret = ret.filter({ $0.offset >= timestamp_begin })
        }
        return ret
    }

    func log_softmax(_ item: [(offset: Int, element: Float)]) -> [(offset: Int, element: Float)] {
        let max_x = item.max(by: { $0.element < $1.element })?.element ?? -Float.infinity
        let sum_exp = item.map({ exp($0.element - max_x) }).reduce(0.0, +)
        return item.map({ (offset: $0.offset, element: log(exp($0.element - max_x)/sum_exp)) })
    }

    func logsumexp(_ item: [Float]) -> Float {
        let max_a = item.max() ?? -Float.infinity
        return max_a + log(item.map({ exp($0 - max_a) }).reduce(0.0, +))
    }

    func getLastToken(logits: MLShapedArray<Float>, tokens: [Int], t_idx: Int = 0) -> Int {
        logits.withUnsafeShapedBufferPointer { ptr, shape, stride in
            let last_token = shape[1] - 1
            var logits = (0..<shape[2]).map({ ptr[stride[1]*last_token + $0] })
                .enumerated().map{ (offset: $0.offset, element: $0.element ) }
            logits = ApplyTimestampRules(tokens: tokens, logits: logits)
            
            guard let max_logit = logits.max(by: { $0.element < $1.element }) else {
                return -1
            }
            return max_logit.offset
        }
    }

    func decodeTokens(tokens: [Int]) -> String {
        let tokenstr = tokens.filter({ $0 < decoder.count }).map({ decoder[$0] }).joined()
        let values = Data(tokenstr.map({ UInt8(codemap[$0] ?? 0) }))
        return String(data: values, encoding: .utf8) ?? ""
    }
}
