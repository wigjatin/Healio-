import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model & tokenizer with proper configuration
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Configure tokenizer settings
tokenizer.pad_token = tokenizer.eos_token  # Set pad token explicitly

# Crisis keywords
crisis_keywords = [
    "suicide", "kill myself", "i want to die", "end my life",
    "i want to end it all", "no reason to live", "depressed",
    "mental breakdown", "life is pointless", "can't go on", "give up"
]

def get_crisis_response():
    return (
        "I'm really sorry you're feeling this way. You're not alone, and there are people who truly care.\n"
        "Please talk to a professional or call a mental health helpline in your country.\n\n"
        "🇮🇳 India: iCall – +91 9152987821\n"
        "🇺🇸 US: Call or text 988\n"
        "🇬🇧 UK: Samaritans at 116 123\n\n"
        "You matter. I'm here to talk as long as you want. 💛"
    )

def generate_reply(prompt, history=None):
    if history is None:
        history = []

    # Check for crisis keywords in current message
    if any(keyword in prompt.lower() for keyword in crisis_keywords):
        return get_crisis_response()

    # Format conversation history
    formatted_history = []
    for i, text in enumerate(history):
        role = "User" if i % 2 == 0 else "Bot"
        formatted_history.append(f"{role}: {text}")
    
    # Create context window (last 3 exchanges)
    context = "\n".join(formatted_history[-6:])
    input_text = f"{context}\nUser: {prompt}\nBot:"
    
    # Tokenize with proper attention mask
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    # Generate response with attention mask
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=200,
            temperature=0.85,
            top_k=100,
            top_p=0.95,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            num_return_sequences=1
        )
    
    # Decode and clean response
    full_response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    
    # Extract bot's response after last "Bot:"
    try:
        reply = full_response.split("Bot:")[-1].strip()
        reply = reply.split("User:")[0].strip()
    except IndexError:
        reply = full_response.strip()
    
    # Final cleanup and fallback
    reply = reply.replace("\n", " ").strip()
    if not reply or len(reply) < 2:
        return "I'm here to listen. Could you tell me more about how you're feeling?"
    
    return reply