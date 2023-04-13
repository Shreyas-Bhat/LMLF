#!/bin/bash

if [ "$#" -eq 0 ]
then
	echo "No input SMILES file passed!"
    echo "USAGE: bash $0 <SMILES_file> <enter>"
    exit 1
fi

if [ "$1" == "--h" ] 
then
    echo "USAGE: bash $0 <SMILES_file> <enter>"
    exit 1
fi

#input="smiles_total_4.txt"; (better to take as an argument)
input=$1

mkdir -p mol2;
mkdir -p pl;

rm -f smiles/$input.smi;
rm -f mol2/$input.mol2;
rm -f pl/$input.pl;

echo "********** PROCESSING FILE: smiles/$input **********";

echo "Extracting SMILES ...";
cat smiles/$input | gawk -F'$' '{print $1}' > /tmp/smi;
cnt=0;
cat /tmp/smi | while read line; 
do 
    cnt=`expr $cnt + 1`; 
    echo -e "$line\tm$cnt" >> smiles/$input.smi;
done

echo "Creating MOL2 ...";
babel -ismi smiles/$input.smi -omol2 mol2/$input.mol2;

echo "Creating PROLOG facts ...";
cat mol2/$input.mol2 | ./newmol2pl A > pl/$input.pl;

echo "Creating class PL facts ...";
cnt=0;
cat smiles/$input | while IFS='$' read f1 f2; 
do
	cnt=`expr $cnt + 1`
	echo "class('m$cnt',$f2)." >> pl/$input.pl;
done;

echo "Generate BK facts (groups and rings) ...";
cp pl/$input.pl tmp/$input.pl;
yap <<+
:- consult(main).
:- gen.
+
mv two_dim.pl pl/$input\_two_dim.pl;
rm tmp/$input.pl;

mkdir -p smi_pp
rm -f smi_pp/$input.*

echo "Preparing input for SMILES++ ..."
cnt=0;
cat smiles/$input | while IFS='$' read f1 f2; 
do 
	cnt=`expr $cnt + 1`; 
	echo "smiles(m$cnt,'$f1')." >> smi_pp/$input.in;
	#echo "mol(m$cnt,m$cnt)." >> smi_pp/$input.in;
	echo "$f2" >> smi_pp/$input.y;
done;

echo "Generating SMILES++ strings ...";
rm -f input.pl two_dim.pl;
ln -s smi_pp/$input.in input.pl;
ln -s pl/$input\_two_dim.pl two_dim.pl;
yap <<+
:- consult(smilespp).
:- tell('output.txt').
:- write_smilespp(false).
:- told.
+
paste -d"$" output.txt smi_pp/$input.y > smi_pp/$input.smipp;
mv output.txt smi_pp/$input\_output.txt;
rm input.pl two_dim.pl;

echo "DONE."
