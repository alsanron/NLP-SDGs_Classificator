#!/bin/bash -x 
# transforms a csv file into asccii files
input=$1
output=$2

for files in $(cd $input ; ls *.pdf); do
   
   name=$(echo $files | cut -f1 -d ".")
   ext=$(echo $files | cut -f2 -d ".")

   filout=$output/${name}.txt

   # Move to text 
   
   pdftotext ${input}/$files $filout

   # Clean it

   perl -pe 's/[^[:ascii:]]//g' $filout > tmp; mv tmp $filout # REmove non-ascii characters
done



