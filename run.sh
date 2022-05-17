for MODEL in GRU LSTM; do
for ENC_LAYERS in 1 2 3; do
for DEC_LAYERS in 1 2 3; do
python train.py --seq2seq=${MODEL} --perception \
	--enc_layers=$ENC_LAYERS --dec_layers=$DEC_LAYERS \
	--epochs=10 --epochs_eval=1 \
	>outputs/${MODEL}_enc${ENC_LAYERS}_dec${DEC_LAYERS}.log
done
done
done