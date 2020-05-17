dir=modified_new
mkdir -p $dir
for file in ./dstat*.csv; do
 file="$(basename "$file")"
 echo $file 
 awk '/^09-10/' $file | awk '/python3/'> ./$dir/$file
 python3 header_mem_csv.py ./$dir/$file
done
