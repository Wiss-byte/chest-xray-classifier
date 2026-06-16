import numpy as np
import torch
import torch.nn as nn

class ManualConvolution2D:
    """Implémentation manuelle de la convolution 2D"""
    
    @staticmethod
    def convolve_2d(image, kernel, padding=0, stride=1):
        """
        Effectue une convolution 2D manuelle
        
        Args:
            image : array 2D (H x W)
            kernel : array 2D (K x K) 
            padding : nombre de pixels à ajouter
            stride : pas de déplacement du kernel
        
        Returns:
            output : résultat de la convolution
        """
        H, W = image.shape
        K, _ = kernel.shape
        
        # Ajouter le padding
        if padding > 0:
            image_padded = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
        else:
            image_padded = image
        
        # Calculer dimensions de sortie
        H_out = (image_padded.shape[0] - K) // stride + 1
        W_out = (image_padded.shape[1] - K) // stride + 1
        
        output = np.zeros((H_out, W_out))
        
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                window = image_padded[h_start:h_start+K, w_start:w_start+K]
                output[i, j] = np.sum(window * kernel)
        
        return output
    
    @staticmethod
    def print_convolution_example():
        """Affiche un exemple de convolution 2D"""
        print("\n" + "="*60)
        print("EXEMPLE DE CONVOLUTION 2D MANUELLE")
        print("="*60)
        
        # Image 5x5
        image = np.array([
            [1, 2, 3, 0, 1],
            [0, 1, 2, 1, 1],
            [2, 1, 0, 2, 1],
            [1, 0, 1, 2, 0],
            [0, 2, 1, 0, 1]
        ], dtype=float)
        
        # Kernel Sobel (détection de contours)
        kernel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=float)
        
        print("\n📸 Image (5x5):")
        print(image)
        
        print("\n🔧 Kernel Sobel (3x3):")
        print(kernel)
        
        # Convolution sans padding
        output = ManualConvolution2D.convolve_2d(image, kernel, padding=0, stride=1)
        print(f"\n📊 Sortie sans padding (stride=1) : {output.shape}")
        print(output)
        
        # Convolution avec padding
        output_padded = ManualConvolution2D.convolve_2d(image, kernel, padding=1, stride=1)
        print(f"\n📊 Sortie avec padding=1 (stride=1) : {output_padded.shape}")
        print(output_padded)
        
        # Convolution avec stride=2
        output_stride = ManualConvolution2D.convolve_2d(image, kernel, padding=0, stride=2)
        print(f"\n📊 Sortie sans padding (stride=2) : {output_stride.shape}")
        print(output_stride)
        
        print("\n" + "="*60 + "\n")


class ManualMaxPooling2D:
    """Implémentation manuelle du max-pooling 2D"""
    
    @staticmethod
    def max_pool_2d(image, pool_size=2, stride=None):
        """
        Effectue un max-pooling 2D manuel
        
        Args:
            image : array 2D (H x W)
            pool_size : taille du pool (pool_size x pool_size)
            stride : pas de déplacement (défaut = pool_size)
        
        Returns:
            output : résultat du pooling
        """
        if stride is None:
            stride = pool_size
        
        H, W = image.shape
        H_out = (H - pool_size) // stride + 1
        W_out = (W - pool_size) // stride + 1
        
        output = np.zeros((H_out, W_out))
        
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                window = image[h_start:h_start+pool_size, w_start:w_start+pool_size]
                output[i, j] = np.max(window)
        
        return output
    
    @staticmethod
    def print_pooling_example():
        """Affiche un exemple de max-pooling"""
        print("\n" + "="*60)
        print("EXEMPLE DE MAX-POOLING 2D MANUEL")
        print("="*60)
        
        image = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=float)
        
        print("\n📸 Image (4x4):")
        print(image)
        
        output = ManualMaxPooling2D.max_pool_2d(image, pool_size=2, stride=2)
        print(f"\n📊 Max-Pooling (2x2, stride=2) : {output.shape}")
        print(output)
        
        output_stride1 = ManualMaxPooling2D.max_pool_2d(image, pool_size=2, stride=1)
        print(f"\n📊 Max-Pooling (2x2, stride=1) : {output_stride1.shape}")
        print(output_stride1)
        
        print("\n" + "="*60 + "\n")


class ManualAvgPooling2D:
    """Implémentation manuelle de l'average pooling 2D"""
    
    @staticmethod
    def avg_pool_2d(image, pool_size=2, stride=None):
        """Effectue un average pooling 2D manuel"""
        if stride is None:
            stride = pool_size
        
        H, W = image.shape
        H_out = (H - pool_size) // stride + 1
        W_out = (W - pool_size) // stride + 1
        
        output = np.zeros((H_out, W_out))
        
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                window = image[h_start:h_start+pool_size, w_start:w_start+pool_size]
                output[i, j] = np.mean(window)
        
        return output
    
    @staticmethod
    def print_pooling_example():
        """Affiche un exemple d'average pooling"""
        print("\n" + "="*60)
        print("EXEMPLE D'AVERAGE POOLING 2D MANUEL")
        print("="*60)
        
        image = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=float)
        
        print("\n📸 Image (4x4):")
        print(image)
        
        output = ManualAvgPooling2D.avg_pool_2d(image, pool_size=2, stride=2)
        print(f"\n📊 Average-Pooling (2x2, stride=2) : {output.shape}")
        print(output)
        
        print("\n" + "="*60 + "\n")


class ConvolutionDimensionCalculator:
    """Calcule les dimensions en convolution"""
    
    @staticmethod
    def calculate_output_size(input_size, kernel_size, padding=0, stride=1, dilation=1):
        """
        Calcule la taille de sortie en convolution
        Formule : H_out = floor((H_in + 2*padding - dilation*(K-1) - 1) / stride + 1)
        """
        output = np.floor(
            (input_size + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1
        )
        return int(output)
    
    @staticmethod
    def print_dimension_table():
        """Affiche un tableau des dimensions"""
        print("\n" + "="*80)
        print("TABLEAU DES DIMENSIONS DE SORTIE EN CONVOLUTION")
        print("="*80)
        
        input_size = 224  # Taille typique
        kernel_size = 3
        
        print(f"\nInput Size: {input_size}x{input_size}, Kernel: {kernel_size}x{kernel_size}\n")
        print(f"{'Padding':<10} {'Stride':<10} {'Dilation':<10} {'Output Size':<15}")
        print("-" * 50)
        
        for padding in [0, 1, 2]:
            for stride in [1, 2]:
                for dilation in [1, 2]:
                    output = ConvolutionDimensionCalculator.calculate_output_size(
                        input_size, kernel_size, padding, stride, dilation
                    )
                    print(f"{padding:<10} {stride:<10} {dilation:<10} {output}x{output:<11}")
        
        print("\n" + "="*80 + "\n")


def demonstrate_all():
    """Démontre tous les concepts"""
    ManualConvolution2D.print_convolution_example()
    ManualMaxPooling2D.print_pooling_example()
    ManualAvgPooling2D.print_pooling_example()
    ConvolutionDimensionCalculator.print_dimension_table()


if __name__ == "__main__":
    demonstrate_all()