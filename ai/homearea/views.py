from django.shortcuts import render
import torch
from django.http import JsonResponse
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# Create your views here.
def chat(request):
    try:
        if request.method == "POST":
            user_input = request.POST.get("input_text")
            model_name = "gpt2"
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)
            input_ids = tokenizer.encode(user_input, return_tensors="pt")
            output = model.generate(input_ids, max_length=50, num_return_sequences=1)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            return JsonResponse({"response": response})

        return render(request, "chat.html")
    except Exception as e:
        return JsonResponse({"error": f"{e}", "success": False})
