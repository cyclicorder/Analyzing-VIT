import argparse
import torch
from PGD_Embedding_Optimizer import PGDEmbeddingOptimizer
from helper import load_image, save_image_file, inverse_normalize
from transformers import ViTForImageClassification, AutoImageProcessor

def main():
    parser = argparse.ArgumentParser(description="Optimize Single Image Embedding through PGD")
    parser.add_argument("--current_image_path", type=str, required=True, help="File path of the current image")
    parser.add_argument("--target_image_path", type=str, required=True, help="File path of the target image to match")
    parser.add_argument("--learning_rate", type=float, default=0.03, help="Learning rate for gradient descent")
    parser.add_argument("--epsilon", type=float, default=0.1, help="maximal allowed input change")
    parser.add_argument("--l2_dist_threshold", type=float, default=1e-4, help="Squared L2 distance threshold")
    parser.add_argument("--cosine_sim_threshold", type=float, default=0.98, help="File path of the current image")
    parser.add_argument("--output_path", type=str, required=True, help="File path to save the optimized image")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = "google/vit-base-patch16-384"
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = ViTForImageClassification.from_pretrained(checkpoint).to(device)

    optimizer = PGDEmbeddingOptimizer(model, args.learning_rate)
    current_image = load_image(args.current_image_path, processor, device)
    target_image = load_image(args.target_image_path, processor, device)

    with torch.no_grad():
        outputs = model(pixel_values=target_image, output_hidden_states=True)
        target_image_emb = outputs.hidden_states[-1][0, 0]

    save_image_file(inverse_normalize()(current_image), args.output_path, "preprocessed_input_image")
    optimized_image, _, _, _, _, _, _, _ = optimizer.optimize_embeddings_pgd(
        current_image, target_image_emb, args.epsilon, args.l2_dist_threshold, args.cosine_sim_threshold
    )
    optimized_image_inv = inverse_normalize()(optimized_image)
    save_image_file(optimized_image_inv, args.output_path, "optimized_image_pgd")

if __name__ == '__main__':
    main()
