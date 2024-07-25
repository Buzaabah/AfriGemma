from transformers import T5ForConditionalGeneration, MT5Model, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM



# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("LLaMAX/LLaMAX3-8B-Alpaca", cache_dir=".cache")
AutoModelForCausalLM.from_pretrained("LLaMAX/LLaMAX3-8B-Alpaca", cache_dir=".cache")


tokenizer = AutoTokenizer.from_pretrained("LLaMAX/LLaMAX2-7B-Alpaca", cache_dir=".cache")
AutoModelForCausalLM.from_pretrained("LLaMAX/LLaMAX2-7B-Alpaca", cache_dir=".cache")


tokenizer = AutoTokenizer.from_pretrained("LLaMAX/LLaMAX2-7B-MetaMath", cache_dir=".cache")
AutoModelForCausalLM.from_pretrained("LLaMAX/LLaMAX2-7B-MetaMath", cache_dir=".cache")

#model = T5ForConditionalGeneration.from_pretrained('google/byt5-xxl')
#tokenizer = AutoTokenizer.from_pretrained('google/byt5-xxl')


#model = MT5Model.from_pretrained('google/mt5-base')
#tokenizer = AutoTokenizer.from_pretrained('google/mt5-base')

#AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf", cache_dir=".cache")
#AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf", cache_dir=".cache")