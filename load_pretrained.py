from speechbrain.inference.ASR import EncoderDecoderASR

asr_model = EncoderDecoderASR.from_hparams(source="/scratch/mfron/IST-LAB/results/conformer_small/7775/save/CKPT+2025-03-26+00-23-25+00", 
                                           hparams_file='conformer_small.yaml', 
                                           savedir="./model.ckpt")
asr_model.transcribe_file('/scratch/mfron/IST-LAB/JASMIN/kaldi_processed/nl_test_read_all_hires/data/format.7/N000172-fn000418-371.8139-373.6982.wav')