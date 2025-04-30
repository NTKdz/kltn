const fs = require('fs');
const crypto = require('crypto');

function simpleLoopTest() {
    console.log("\n=== Test 1: Simple Loop ===");
    let start = Date.now();
    let sum = 0;
    for (let i = 0; i < 1_000_000_000; i++) {
        sum += i;
    }
    let end = Date.now();
    console.log("Sum:", sum);
    console.log("Time:", (end - start), "ms");
}

function arraySortTest() {
    console.log("\n=== Test 2: Array Sort ===");
    let size = 1_000_000;
    let arr = Array.from({ length: size }, () => Math.floor(Math.random() * 1_000_000));
    let start = Date.now();
    arr.sort((a, b) => a - b);
    let end = Date.now();
    console.log("Sort time:", (end - start), "ms");
}

function recursionTest() {
    console.log("\n=== Test 3: Recursion (Fibonacci 30) ===");
    function fib(n) {
        if (n <= 1) return n;
        return fib(n - 1) + fib(n - 2);
    }
    let start = Date.now();
    let result = fib(30);
    let end = Date.now();
    console.log("fib(30) =", result);
    console.log("Time:", (end - start), "ms");
}

function stringConcatTest() {
    console.log("\n=== Test 4: String Concatenation ===");
    let start = Date.now();
    let result = '';
    for (let i = 0; i < 1_000_000; i++) {
        result += i;
    }
    let end = Date.now();
    console.log("Result length:", result.length);
    console.log("Time:", (end - start), "ms");
}

function fileWriteTest() {
    console.log("\n=== Test 5: Write 100 MB File ===");
    let start = Date.now();
    let data = '';
    for (let i = 0; i < 10_000_000; i++) {
        data += "abcdefghij\n";
    }
    fs.writeFileSync('output_node.txt', data);
    let end = Date.now();
    console.log("File write time:", (end - start), "ms");
}

function hashingTest() {
    console.log("\n=== Test 6: SHA-256 Hashing 1 Million Times ===");
    let start = Date.now();
    for (let i = 0; i < 1_000_000; i++) {
        crypto.createHash('sha256').update('testdata').digest('hex');
    }
    let end = Date.now();
    console.log("Hashing time:", (end - start), "ms");
}

function matrixMultiplicationTest() {
    console.log("\n=== Test 7: 500x500 Matrix Multiplication ===");
    let size = 500;
    let A = Array.from({length: size}, () => Array.from({length: size}, () => Math.floor(Math.random() * 100)));
    let B = Array.from({length: size}, () => Array.from({length: size}, () => Math.floor(Math.random() * 100)));
    let C = Array.from({length: size}, () => Array(size).fill(0));

    let start = Date.now();
    for (let i = 0; i < size; i++)
        for (let j = 0; j < size; j++)
            for (let k = 0; k < size; k++)
                C[i][j] += A[i][k] * B[k][j];
    let end = Date.now();
    console.log("Matrix multiplication time:", (end - start), "ms");
}

function runAllTests() {
    simpleLoopTest();
    arraySortTest();
    recursionTest();
    stringConcatTest();
    fileWriteTest();
    hashingTest();
    matrixMultiplicationTest();
}

runAllTests();
