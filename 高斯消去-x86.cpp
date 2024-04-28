#include<iostream>
#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include <xmmintrin.h>
#include <immintrin.h>
using namespace std;
const int MAX_SIZE = 4000;
const int size_[13] = { 10,20,30,50,100,200,400,800,1000,1500,2000,3000,4000 };
float A[MAX_SIZE][MAX_SIZE];
void initial(int N) {
    srand((int)time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = 1 + rand() % 100;
        }
    }
}
void normal(int N);
void normal_sse(int N);
void normal_sse_al(int N);
void normal_avx(int N);
void normal_avx512(int N);

int main() {
    struct timespec sts, ets;
    for (int t = 0; t < 13; t++) {
        int N = size_[t];
        initial(N);
        timespec_get(&sts, TIME_UTC);
        normal(N);
        timespec_get(&ets, TIME_UTC);
        time_t dsec = ets.tv_sec - sts.tv_sec;
        long dnsec = ets.tv_nsec - sts.tv_nsec;
        if (dnsec < 0) {
            dsec--;
            dnsec += 1000000000ll;
        }
        printf("%ld.%09ld\n", dsec, dnsec);
    }
    for (int t = 0; t < 13; t++) {
        int N = size_[t];
        initial(N);
        timespec_get(&sts, TIME_UTC);
        normal_sse(N);
        timespec_get(&ets, TIME_UTC);
        time_t dsec = ets.tv_sec - sts.tv_sec;
        long dnsec = ets.tv_nsec - sts.tv_nsec;
        if (dnsec < 0) {
            dsec--;
            dnsec += 1000000000ll;
        }
        printf("%ld.%09ld\n", dsec, dnsec);
    }
    for (int t = 0; t < 13; t++) {
        int N = size_[t];
        initial(N);
        timespec_get(&sts, TIME_UTC);
        normal_sse_al(N);
        timespec_get(&ets, TIME_UTC);
        time_t dsec = ets.tv_sec - sts.tv_sec;
        long dnsec = ets.tv_nsec - sts.tv_nsec;
        if (dnsec < 0) {
            dsec--;
            dnsec += 1000000000ll;
        }
        printf("%ld.%09ld\n", dsec, dnsec);
    }
    for (int t = 0; t < 13; t++) {
        int N = size_[t];
        initial(N);
        timespec_get(&sts, TIME_UTC);
        normal_avx(N);
        timespec_get(&ets, TIME_UTC);
        time_t dsec = ets.tv_sec - sts.tv_sec;
        long dnsec = ets.tv_nsec - sts.tv_nsec;
        if (dnsec < 0) {
            dsec--;
            dnsec += 1000000000ll;
        }
        printf("%ld.%09ld\n", dsec, dnsec);
    }
    return 0;
}

void normal(int N) {
    for (int k = 0; k < N; k++) {
        for (int j = k + 1; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

void normal_sse(int N) {
    for (int k = 0; k < N; k++) {
        __m128 vt = _mm_set_ps1(A[k][k]);
        int j = k + 1;
        for (; j + 4 <= N; j += 4) {
            __m128 va = _mm_loadu_ps(&A[k][j]);
            va = _mm_div_ps(va, vt);
            _mm_storeu_ps(&A[k][j], va);
        }
        for (; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            __m128 vaik = _mm_set_ps1(A[i][k]);
            int t = k + 1;
            for (; t + 4 <= N; t += 4) {
                __m128 vakj = _mm_loadu_ps(&A[k][t]);
                __m128 vaij = _mm_loadu_ps(&A[i][t]);
                __m128 vx = _mm_mul_ps(vakj, vaik);
                vaij = _mm_sub_ps(vaij, vx);
                _mm_storeu_ps(&A[i][t], vaij);
            }
            for (; t < N; t++) {
                A[i][t] -= A[i][k] * A[k][t];
            }
            A[i][k] = 0.0;
        }
    }
}

void normal_sse_al(int N) {
    for (int k = 0; k < N; k++) {
        __m128 vt = _mm_set_ps1(A[k][k]);
        int j = k + 1;
        int off = j % 4;
        for (int i = 0; i < 4 - off; i++) {
            A[k][j + i] /= A[k][k];
        }
        j = j + 4 - off;
        for (; j + 4 <= N; j += 4) {
            __m128 va = _mm_load_ps(&A[k][j]);
            va = _mm_div_ps(va, vt);
            _mm_storeu_ps(&A[k][j], va);
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            __m128 vaik = _mm_set_ps1(A[i][k]);
            int t = k + 1;
            int off1 = t % 4;
            for (int p = 0; p < 4 - off1; p++) {
                A[i][t + p] -= A[i][k] * A[k][t + p];
            }
            t = t + 4 - off1;
            for (; t + 4 <= N; t += 4) {
                __m128 vakj = _mm_load_ps(&A[k][t]);
                __m128 vaij = _mm_load_ps(&A[i][t]);
                __m128 vx = _mm_mul_ps(vakj, vaik);
                vaij = _mm_sub_ps(vaij, vx);
                _mm_store_ps(&A[i][t], vaij);
            }
            A[i][k] = 0.0;
        }
    }
}

void normal_avx(int N) {
    for (int k = 0; k < N; k++) {
        __m256 vt = _mm256_set1_ps(A[k][k]);
        int j = k + 1;
        for (; j + 8 <= N; j += 8) {
            __m256 va = _mm256_loadu_ps(&A[k][j]);
            va = _mm256_div_ps(va, vt);
            _mm256_storeu_ps(&A[k][j], va);
        }
        for (; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            __m256 vaik = _mm256_set1_ps(A[i][k]);
            int t = k + 1;
            for (; t + 8 <= N; t += 8) {
                __m256 vakj = _mm256_loadu_ps(&A[k][t]);
                __m256 vaij = _mm256_loadu_ps(&A[i][t]);
                __m256 vx = _mm256_mul_ps(vakj, vaik);
                vaij = _mm256_sub_ps(vaij, vx);
                _mm256_storeu_ps(&A[i][t], vaij);
            }
            for (; t < N; t++) {
                A[i][t] -= A[i][k] * A[k][t];
            }
            A[i][k] = 0.0;
        }
    }
}

void normal_avx512(int N) {
    for (int k = 0; k < N; k++) {
        __m512 vt = _mm512_set1_ps(A[k][k]);
        int j = k + 1;
        for (; j + 16 <= N; j += 16) {
            __m512 va = _mm512_loadu_ps(&A[k][j]);
            va = _mm512_div_ps(va, vt);
            _mm512_storeu_ps(&A[k][j], va);
        }
        for (; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            __m512 vaik = _mm512_set1_ps(A[i][k]);
            int t = k + 1;
            for (; t + 16 <= N; t += 16) {
                __m512 vakj = _mm512_loadu_ps(&A[k][t]);
                __m512 vaij = _mm512_loadu_ps(&A[i][t]);
                __m512 vx = _mm512_mul_ps(vakj, vaik);
                vaij = _mm512_sub_ps(vaij, vx);
                _mm512_storeu_ps(&A[i][t], vaij);
            }
            for (; t < N; t++) {
                A[i][t] -= A[i][k] * A[k][t];
            }
            A[i][k] = 0.0;
        }
    }
}