#include<iostream>
#include<time.h>
#include<stdio.h>
#include <arm_neon.h>
#include<stdlib.h>
using namespace std;
const int size[13] = { 10,20,30,50,100,200,400,800,1000,1500,2000,3000,4000 };
float A[4000][4000];
void initial(int N) {
    srand((int)time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = 1 + rand() % 100;
        }
    }
}

void gaosi(int N) {
    float32x4_t vt = vdupq_n_f32(0.0);
    float32x4_t va = vdupq_n_f32(0.0);
    float32x4_t vaik = vdupq_n_f32(0.0);
    float32x4_t vakj = vdupq_n_f32(0.0);
    float32x4_t vaij = vdupq_n_f32(0.0);
    float32x4_t vx = vdupq_n_f32(0.0);
    for (int k = 0; k < N; k++) {
        vt = vdupq_n_f32(A[k][k]); // 把数值拷贝四份到寄存器中
        int j = k + 1;
        for (; j + 4 <= N; j += 4) {
            va = vld1q_f32(&A[k][j]);
            va = vdivq_f32(va, vt);
            vst1q_f32(&A[k][j], va);
        }
        for (; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            vaik = vdupq_n_f32(A[i][k]);
            int t = k + 1;
            for (; t + 4 <= N; t += 4) {
                vakj = vld1q_f32(&A[k][t]);
                vaij = vld1q_f32(&A[i][t]);
                vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&A[i][t], vaij);
            }
            for (; t < N; t++) {
                A[i][t] -= A[i][k] * A[k][t];
            }
            A[i][k] = 0;
        }
    }
}
void gaosi1(int N) {
    for (int k = 0; k < N; k++) {
        float32x4_t vt = vdupq_n_f32(A[k][k]); // 把数值拷贝四份到寄存器中
        int j = k + 1;
        for (; j + 4 <= N; j += 4) {
            float32x4_t va = vld1q_f32(&A[k][j]);
            va = vdivq_f32(va, vt);
            vst1q_f32(&A[k][j], va);
        }
        for (; j < N; j++) {
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


void gaosi2(int N) {
    for (int k = 0; k < N; k++) {
        for (int j = k + 1; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            float32x4_t vaik = vdupq_n_f32(A[i][k]);
            int t = k + 1;
            for (; t + 4 <= N; t += 4) {
                float32x4_t vakj = vld1q_f32(&A[k][t]);
                float32x4_t vaij = vld1q_f32(&A[i][t]);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&A[i][t], vaij);
            }
            for (; t < N; t++) {
                A[i][t] -= A[i][k] * A[k][t];
            }
            A[i][k] = 0;
        }
    }
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




int main() {
    struct timespec sts, ets;
    for (int t = 0; t < 13; t++) {
        int N = size[t];
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
        int N = size[t];
        initial(N);
        timespec_get(&sts, TIME_UTC);
        gaosi(N);
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
        int N = size[t];
        initial(N);
        timespec_get(&sts, TIME_UTC);
        gaosi1(N);
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
        int N = size[t];
        initial(N);
        timespec_get(&sts, TIME_UTC);
        gaosi2(N);
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
