for MODEL in GRU LSTM; do
for LAYERS in 1 2 3; do
for HID_DIM in 128 256 512; do
for DROPOUT in 0.1 0.5; do
python train.py --seq2seq=${MODEL} --perception \
	--enc_layers=$LAYERS --dec_layers=$LAYERS \
	--epochs=1 --epochs_eval=1 --dropout=$DROPOUT \
	>outputs/${MODEL}_layers${LAYERS}_hid${HID_DIM}_drop${DROPOUT}.log
done
done
done
done