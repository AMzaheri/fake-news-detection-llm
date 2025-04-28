#-------------------------------------------------
#This script downloads and loads a pretrained DistilBERT model from Hugging Face.
#After running : Hugging Face gives you a heads-up that installing hf_xet can speed up future downloads (optional).
# We pass a news article through the model, and it predicts "REAL" or "False".
#-------------------------------------------------
#from llm_model_module import predict_with_llm

#print(predict_with_llm("Aliens have landed in London and started a rock band."))
#print(predict_with_llm("The economy showed signs of recovery in the second quarter."))

from llm_model_module import prepare_dataset, tokenize_dataset

# Step 1: Prepare the dataset
dataset = prepare_dataset()  # This returns a Hugging Face DatasetDict
#print(dataset)  # Should now show DatasetDict with train/test
#print(dataset['train'][0])  # Should work without error

# Step 2: Tokenize the dataset
tokenized_dataset = tokenize_dataset(dataset)

# Print to verify
#print(tokenized_dataset)
#print(tokenized_dataset['train'][0])  # Check the first example

from llm_model_module import train_model
train_model(tokenized_dataset)
