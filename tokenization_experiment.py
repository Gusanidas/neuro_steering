import torch
from transformers import GPT2Tokenizer
import pandas as pd
from rich.console import Console
from rich.table import Table

def main():
    # Load the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Define a sentence to tokenize
    sentence = "Hello, how are you doing today? I was thinking on going to the park to eat some watermelon. The thing about watermelon, is that it is very good for you."
    words = sentence.split()
    
    # Initialize a rich console for pretty output
    console = Console()
    
    # Create a table to display results
    table = Table(title="Tokenization by Letter Addition")
    table.add_column("Text", style="cyan")
    table.add_column("Tokens", style="green")
    table.add_column("Token IDs", style="yellow")
    
    # Add one letter at a time and show tokenization
    for i, word in enumerate(words):
        # Take the sentence up to the i-th character
        current_text = " ".join(words[:i])
        
        # Tokenize the current text
        tokens = tokenizer.tokenize(current_text)
        token_ids = tokenizer.encode(current_text)
        
        # Add a row to the table
        table.add_row(
            current_text,
            str(tokens),
            str(token_ids)
        )
    
    # Display the table
    console.print(table)
    
    # Alternative visualization as CSV
    results = []
    for i in range(1, len(sentence) + 1):
        current_text = sentence[:i]
        tokens = tokenizer.tokenize(current_text)
        token_ids = tokenizer.encode(current_text)
        
        results.append({
            "text": current_text,
            "tokens": tokens,
            "token_ids": token_ids
        })
    
    # Create a DataFrame and save to CSV
    df = pd.DataFrame(results)
    csv_filename = "tokenization_results.csv"
    df.to_csv(csv_filename, index=False)
    console.print(f"\nResults also saved to {csv_filename}")

if __name__ == "__main__":
    main() 