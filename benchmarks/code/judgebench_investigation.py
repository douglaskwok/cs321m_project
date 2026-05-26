from datasets import load_dataset
import pandas as pd

ds = load_dataset("mattymchen/codejudgebench", "codegen")
df = pd.concat([ds[split].to_pandas() for split in ds])
print(df["question_id"].nunique())  # unique LCB questions in codegen

ds2 = load_dataset("mattymchen/codejudgebench", "codegen_pass5")
df2 = pd.concat([ds2[split].to_pandas() for split in ds2])
print(df2["question_id"].nunique())  # unique LCB questions in codegen_pass

ds3 = load_dataset("mattymchen/codejudgebench", "coderepair")
df3 = pd.concat([ds3[split].to_pandas() for split in ds3])
print(df3["question_id"].nunique())  # unique LCB questions in codegen_pass

ds4 = load_dataset("mattymchen/codejudgebench", "testgen")
df4 = pd.concat([ds4[split].to_pandas() for split in ds4])
print(df4["question_id"].nunique())  # unique LCB questions in codegen_pass

df5 = pd.concat([df4, df3, df2, df])
print(df5["question_id"].nunique())  # unique LCB questions in codegen_pass

codegen_ids = set(df["question_id"])
coderepair_ids = set(df3["question_id"])
testgen_ids = set(df4["question_id"])

print(len(codegen_ids & coderepair_ids))   # in both codegen and coderepair
print(len(codegen_ids & testgen_ids))       # in both codegen and testgen
print(len(coderepair_ids & testgen_ids))    # in both coderepair and testgen
print(len(codegen_ids & coderepair_ids & testgen_ids))  # in all three
