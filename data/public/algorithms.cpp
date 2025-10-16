/**
 * InnovateX Core Algorithms
 * 
 * This file contains core algorithms used in the InnovateX platform
 * for data processing and optimization.
 * 
 * @author InnovateX Development Team
 * @version 1.0
 * @date 2024-01-15
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

namespace InnovateX {
    
    /**
     * QuickSort implementation for efficient data sorting
     * Time Complexity: O(n log n) average case
     * Space Complexity: O(log n)
     */
    template<typename T>
    void quickSort(std::vector<T>& arr, int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);
            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
    }
    
    template<typename T>
    int partition(std::vector<T>& arr, int low, int high) {
        T pivot = arr[high];
        int i = (low - 1);
        
        for (int j = low; j <= high - 1; j++) {
            if (arr[j] < pivot) {
                i++;
                std::swap(arr[i], arr[j]);
            }
        }
        std::swap(arr[i + 1], arr[high]);
        return (i + 1);
    }
    
    /**
     * Binary search implementation for fast data retrieval
     * Time Complexity: O(log n)
     * Space Complexity: O(1)
     */
    template<typename T>
    int binarySearch(const std::vector<T>& arr, T target) {
        int left = 0;
        int right = arr.size() - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (arr[mid] == target) {
                return mid;
            }
            
            if (arr[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return -1; // Element not found
    }
    
    /**
     * Calculate moving average for time series data
     * Used in our analytics platform
     */
    std::vector<double> movingAverage(const std::vector<double>& data, int windowSize) {
        std::vector<double> result;
        
        if (data.size() < windowSize) {
            return result;
        }
        
        for (size_t i = windowSize - 1; i < data.size(); i++) {
            double sum = 0.0;
            for (int j = 0; j < windowSize; j++) {
                sum += data[i - j];
            }
            result.push_back(sum / windowSize);
        }
        
        return result;
    }
    
    /**
     * Euclidean distance calculation for ML algorithms
     */
    double euclideanDistance(const std::vector<double>& point1, 
                           const std::vector<double>& point2) {
        if (point1.size() != point2.size()) {
            throw std::invalid_argument("Points must have same dimensions");
        }
        
        double sum = 0.0;
        for (size_t i = 0; i < point1.size(); i++) {
            double diff = point1[i] - point2[i];
            sum += diff * diff;
        }
        
        return std::sqrt(sum);
    }
}

// Example usage
int main() {
    std::vector<int> data = {64, 34, 25, 12, 22, 11, 90};
    
    std::cout << "Original array: ";
    for (int x : data) {
        std::cout << x << " ";
    }
    std::cout << std::endl;
    
    InnovateX::quickSort(data, 0, data.size() - 1);
    
    std::cout << "Sorted array: ";
    for (int x : data) {
        std::cout << x << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
