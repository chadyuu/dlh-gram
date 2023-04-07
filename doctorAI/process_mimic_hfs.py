# This script processes MIMIC-III dataset and builds longitudinal diagnosis records for patients with at least two visits.
# The output data are cPickled, and suitable for training Doctor AI or RETAIN
# Written by Edward Choi (mp2893@gatech.edu)
# Usage: Put this script to the foler where MIMIC-III CSV files are located. Then execute the below command.
# python process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv <output file> 

#### -------- Note from ALLAN: this was converted to Python 3+ --------
# Replaced arguments with the values so I can debug; you can just type 'python process_mimic.py' in terminal
# python process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv output/output

# Output files
# <output file>.pids: List of unique Patient IDs. Used for intermediate processing
# <output file>.dates: List of List of Python datetime objects. The outer List is for each patient. The inner List is for each visit made by each patient
# <output file>.seqs: List of List of List of integer diagnosis codes. The outer List is for each patient. The middle List contains visits made by each patient. The inner List contains the integer diagnosis codes that occurred in each visit
# <output file>.types: Python dictionary that maps string diagnosis codes to integer diagnosis codes.

import sys
import pickle as pickle
from datetime import datetime

def convert_to_icd9(dxStr):
	if dxStr.startswith('E'):
		if len(dxStr) > 4: return dxStr[:4] + '.' + dxStr[4:]
		else: return dxStr
	else:
		if len(dxStr) > 3: return dxStr[:3] + '.' + dxStr[3:]
		else: return dxStr
	
def convert_to_3digit_icd9(dxStr):
	if dxStr.startswith('E'):
		if len(dxStr) > 4: return dxStr[:4]
		else: return dxStr
	else:
		if len(dxStr) > 3: return dxStr[:3]
		else: return dxStr

if __name__ == '__main__':
	admissionFile = "ADMISSIONS.csv"			# sys.argv[1]	# edited by Allan
	diagnosisFile = "DIAGNOSES_ICD.csv"			# sys.argv[2]
	outFile = 		"output/output"				# sys.argv[3]

	print('Building pid-admission mapping, admission-date mapping')
	pidAdmMap = {}
	admDateMap = {}
	infd = open(admissionFile, 'r')
	infd.readline()
	for line in infd:
		tokens = line.strip().split(',')
		pid = int(tokens[1])
		admId = int(tokens[2])
		admTime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
		admDateMap[admId] = admTime
		if pid in pidAdmMap: pidAdmMap[pid].append(admId)
		else: pidAdmMap[pid] = [admId]
	infd.close()

	print('Building admission-dxList mapping')
	admDxMap = {}
	infd = open(diagnosisFile, 'r')
	infd.readline()
	for line in infd:
		tokens = line.strip().split(',')
		admId = int(tokens[2])
		dxStr = 'D_' + convert_to_icd9(tokens[4][1:-1]) ############## Uncomment this line and comment the line below, if you want to use the entire ICD9 digits.
		#dxStr = 'D_' + convert_to_3digit_icd9(tokens[4][1:-1])
		if admId in admDxMap: admDxMap[admId].append(dxStr)
		else: admDxMap[admId] = [dxStr]
	infd.close()

	print('Building pid-sortedVisits mapping')
	pidSeqMap = {}
	for pid, admIdList in pidAdmMap.items():
		if len(admIdList) < 2: continue
		sortedList = sorted([(admDateMap[admId], admDxMap[admId]) for admId in admIdList])
		pidSeqMap[pid] = sortedList
	
	print('Building pids, dates, strSeqs')
	pids = []
	dates = []
	seqs = []
	for pid, visits in pidSeqMap.items():
		pids.append(pid)
		seq = []
		date = []
		for visit in visits:
			date.append(visit[0])
			seq.append(visit[1])
		dates.append(date)
		seqs.append(seq)
	
	print('Converting strSeqs to intSeqs, and making types')
	types = {}
	newSeqs = []
	hf_labels = []  	# by allan
	for patient in seqs:
		newPatient = []
		hf_label = 0 	# by allan
		for visit in patient:
			newVisit = []
			for code in visit:
				
				# 'if' added by allan.  filter out HF diag using ICD-9 codes starting w/ D_428
				if code.startswith('D_428'):
					hf_label = 1  
					break # omit all visits after HF

				if code in types:
					newVisit.append(types[code])
				else:
					types[code] = len(types)
					newVisit.append(types[code])
			if hf_label == 1: # by allan
				break # omit all the visits after heart failure
			newPatient.append(newVisit)
		newSeqs.append(newPatient)
		hf_labels.append(hf_label)  # by allan


	
	print("len(pids): ", len(pids))  	# 7537   # patient id's
	print("len(dates): ", len(dates))	# 7537	 # datetime objects
	print("len(seqs): ", len(newSeqs))	# 7537	 # per patient, their list of visits (list of lists)
	print("len(types): ", len(types))	# 4894  This is the # of unique medical codes
	print("len(hf_labels): ", len(hf_labels))  # 7537   # of hf labels (0 or 1 values)
	pickle.dump(pids, open(outFile+'.pids', 'wb'), -1)
	pickle.dump(dates, open(outFile+'.dates', 'wb'), -1)
	pickle.dump(newSeqs, open(outFile+'.seqs', 'wb'), -1)
	pickle.dump(types, open(outFile+'.types', 'wb'), -1)
 
	pickle.dump(hf_labels, open(outFile+'.hfs', 'wb'), -1)
