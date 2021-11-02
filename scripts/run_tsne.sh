for m in 0 1 2
do
for d in 0 1
do
for b in 15 30
do
for n in 1 2
do

python src/vis_tSNE.py -b $b -d $d -m $m -n $n

done
done
done
done