import openai
import jsonlines
import time
import pandas as pd
import numpy as np
import random
import json
import sys
import os
from rdkit.Chem import Descriptors, QED
from rdkit.Chem import rdMolDescriptors as rdmd

#{"label": "1", "smiles": "O=c1[nH]c2cnc(-n3cnc4ccc(F)cc43)nc2n1Cc1cccc(F)c1F"}
# Set up your API key and model parameters
#openai.api_key = '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'
openai.api_key = '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'\
#openai.apy_key = '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'
#model_engine = 'gpt-3.5-turbo' # You can choose a different model if desired
model_engine = 'text-davinci-002'
temperature = 0.3 
num_generations = 10# The number of times to generate new molecules and feed them back into the modelcond
import openai
import jsonlines
import time
import subprocess
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import RDConfig
from rdkit.Chem import QED
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


directory_path = '/home/datalab/BioLLM/'
target_file_path = '/home/datalab/BioLLM/drd2.jsonl'
output_file_path = '/home/datalab/BioLLM/drd2n10nofeedback_data.csv'
target_molecules = []
target_labels = []

threshold_increment_frequency = 2
data = []
def calculate_docking_score(smiles):
    try:
        # Generate an RDKit molecule from the SMILES string
        molecule = Chem.MolFromSmiles(smiles)
        molecule = Chem.AddHs(molecule)
    except Exception as e:
        print(f"SMILES Parse Error: {e}. Skipping molecule: {smiles}")
        return None
    if molecule is not None:
        # Generate 2D coordinates for the molecule
        try:
            AllChem.Compute2DCoords(molecule)
        except Exception as e:
            print(f"Compute2DCoords Error: {e}. Skipping molecule: {smiles}")
            return None

        # Generate a 3D conformation of the molecule using the ETKDG method
        AllChem.EmbedMolecule(molecule, AllChem.ETKDG())

        # Optimize the 3D conformation
        try:
            AllChem.MMFFOptimizeMolecule(molecule)
        except Exception as e:
            print(f"MMFFOptimizeMolecule Error: {e}. Skipping molecule: {smiles}")
            return None

        # Generate a PDB file from the molecule
        pdb_filename = '/home/datalab/BioLLM/ligand.pdb'
        writer = Chem.PDBWriter(pdb_filename)
        writer.write(molecule)
        writer.close()

        # Run the gnina docking script
        #/home/datalab/gnina --config JAK2_config.txt --ligand /home/datalab/BioLLM/chatgpt/${b}.sdf --out /home/datalab/BioLLM/chatgpt_output_sdf/${b}_out.sdf --log /home/datalab/BioLLM/chatgpt_active_output_logs/${b}_log.txt --cpu 4 --num_modes 1

        #!./gnina -r /content/4IVA.pdb -l /content/ligand.smi --autobox_ligand /content/ligand.smi -o docked.sdf --seed 0
        cmd = ['/home/datalab/gnina', '--config', 'JAK2_config.txt', '--ligand', '/home/datalab/BioLLM/ligand.pdb', '--out', 'output.sdf', '--log', '/home/datalab/BioLLM/threshold_output_log.txt', '--cpu', '4', '--num_modes', '1']
        # 
        #cmd = ['/home/datalab/gnina', '-r', '/home/datalab/BioLLM/4IVA.pdb', '-l', '/home/datalab/BioLLM/ligand.pdb', '--autobox_ligand', '/home/datalab/BioLLM/ligand.pdb', '-o', '/content/docked.txt', '--seed', '0']

        print("Docking Command:", ' '.join(cmd))
        # try:
        #     subprocess.run(cmd, check=True)
        # except subprocess.CalledProcessError as e:
        #     print("Docking Error:", e)
        #     return None
        try:
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print("Docking process failed:", e)
            print("Error output:", e.stderr)
            return None


        # subprocess.run(cmd, check=True)
        
        # TODO: Extract and return the docking score from the output files
        import os

        # Iterate over the files in the directory
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):  # Consider only the text files
                file_path = os.path.join(directory_path, filename)
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                    for i, line in enumerate(lines):
                        if 'affinity' in line.lower() and 'cnn' in line.lower():
                            third_next_line_values = lines[i + 3].split()
                            if len(third_next_line_values) >= 4:
                                try:
                                    cnn_affinity = float(third_next_line_values[3].strip())
                                    return cnn_affinity
                                except ValueError:
                                    pass
        
        return None

for i in range(1, num_generations):
    
    
    # if i % threshold_increment_frequency == 0 and i > 0:
    #     threshold += threshold_increment
    
    
    new_molecules = []
    num_molecules = 20
    print("iteration", i)
    with jsonlines.open(target_file_path) as reader:
        for line in reader:
            if "\n" not in line:
                target_molecules.append(line['smiles'])
                target_labels.append(line['label'])
    unique_molecules = set() 
    
    for _ in range(num_molecules):

        target_index = random.randint(0, len(target_molecules) - 1)
        target_mol = target_molecules[target_index]
        target_label = target_labels[target_index]
        print("target, label", target_mol, target_label)
        # message = [{"role":"user", "content":f'Generating only SMILES strings, Generate a novel valid molecule similar to {target_mol} that is {target_label}-class'}]
        # response = openai.ChatCompletion.create(
        #     model = "gpt-3.5-turbo",
        #     messages = message,
        #     max_tokens=60,
        #     temperature=0.7,
        #     n=1,
        #     stop=None,
        #     timeout=60
        # )
        # new_mol = response.choices[0]
        prompt = f'Generate a novel valid molecule similar to {target_mol} that is {target_label}-class and do not generate any English text'
        response = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=60,
            temperature=0.7,
            n=1,
            stop=None,
            timeout=20
        )
        new_mol = response.choices[0].text.strip()
        print("new mol", new_mol)
        try:
            mol = Chem.MolFromSmiles(new_mol)
            if mol is not None and mol not in unique_molecules:
                #sanitized_mol = Chem.SanitizeMol(mol)
                new_molecules.append(mol)
                #print("new molecules", new_mol)
                print("new_molecules", new_molecules)
        except Exception as e:
            print(f"SMILES Parse Error: {e}. Skipping molecule: {new_mol}")
        docking_scores = []
        for mol in new_molecules:
            docking_score = calculate_docking_score(mol)
            print("docking score", docking_score)
            docking_scores.append(docking_score)

        labels = []
        mw_threshold = 700
        logp_threshold = 5
        radscore_threshold = 5
        new_target_molecules = []
        mols = []
        
        

        for mol in new_molecules:
            print("mol", Chem.MolToSmiles(mol))
            try:
                smiles = Chem.MolToSmiles(mol)
                if smiles not in unique_molecules:
                    unique_molecules.add(smiles)
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    sas = sascorer.calculateScore(mol)
                    print(mw, logp, sas, )
                    if (
                        200 <= mw <= mw_threshold
                        and logp <= logp_threshold
                        and sas <= radscore_threshold
                    ):
                        labels.append('1')
                        new_target_molecules.append({'smiles': smiles, 'label': '1'})
                        #mols.append(Chem.MolToSmiles(mol))
                    else:
                        labels.append('0')
                else:
                    print("skipping duplicate molecules")
            except Exception as e:
                print(f"Molecular Property Calculation Error: {e}. Skipping molecule.")
            

        data.extend(list(zip(unique_molecules, labels)))
        
        print("data", data)


        #with jsonlines.open(target_file_path, mode='a') as writer:
            # writer.write('\\n')
            # writer.write_all(new_target_molecules)
        with jsonlines.open(target_file_path, mode='a') as writer:
            # writer.write("\\n")
            for molecule in new_target_molecules:
                writer.write(molecule)
                writer.write('\n')
        # with open(target_file_path, mode='a') as outfile:
        #     for hostDict in target_file_path:
        #         json.dump(hostDict, outfile)
        #         outfile.write('\n')

df = pd.DataFrame(data, columns=['Molecule', 'Label'])
df.to_csv(output_file_path, index=False)
# Print and analyze the results
print(f'generation {i}:')
print('generated molecules:')
print(data)