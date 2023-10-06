import torch
all_sequences = []
for i in range(1, 21):  # Assuming 20 jobs were created
    sequences = torch.load(f'small_dataset/sis_sequences_{i}.pt')
    all_sequences.extend(sequences)

torch.save(all_sequences, 'small_dataset/combined_sis_sequences.pt')
