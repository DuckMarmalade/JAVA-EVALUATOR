

// User function Template for Java
class Solution {
    int missingNumber(int arr[]) {
    // int n = arr.length + 1; // Since one number is missing
    //     int expectedSum = n * (n + 1) / 2;
    //     int actualSum = 0;

    //     for (int num : arr) {
    //         actualSum += num;
    //     }

    //     return expectedSum - actualSum;
    
   ArrayList<Integer> list = new ArrayList<>();
        for(int i =0;i<arr.length;i++){
            list.add(arr[i]);
        }
        for(int i =1;i<=arr.length+1;i++){
            if(!list.contains(i)){
                return i;
            }
        }
        return 0;
    }
}