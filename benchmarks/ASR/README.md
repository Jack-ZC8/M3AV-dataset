# ASR
 - This is a recipe for ESPnet.

### Train your models using ESPnet following this:
1. Git clone the codebase of ESPnet.
```
git clone https://github.com/espnet/espnet.git
```
2. Put M3AV into egs2:
```
cp M3AV-dataset/benchmarks/ASR/M3AV espnet/egs2/
```
3. Run the recipe
```
cd espnet/egs2/M3AV/asr1
./run.sh
```
4. We use soft links for the general code, you can find it under the path of egs2/TEMPLATE/asr1.
