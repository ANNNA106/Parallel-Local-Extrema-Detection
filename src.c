#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <stdbool.h>

#define IDX(x, y, z, nx, ny) ((z) * (nx) * (ny) + (y) * (nx) + (x))

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    // double timeStartTotal = MPI_Wtime();
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 10) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <input_file> PX PY PZ NX NY NZ NC <output_file>\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Parse command-line arguments.
    const char* inputFile  = argv[1];
    int PX = atoi(argv[2]);
    int PY = atoi(argv[3]);
    int PZ = atoi(argv[4]);
    int NX = atoi(argv[5]);
    int NY = atoi(argv[6]);
    int NZ = atoi(argv[7]);
    int NC = atoi(argv[8]);
    const char* outputFile = argv[9];

    // Verify that PX*PY*PZ equals number of MPI processes.
    if (PX * PY * PZ != size) {
        if (rank == 0) {
            fprintf(stderr, "Error: PX*PY*PZ != # MPI processes.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    
    // Check that global dimensions are exactly divisible.
    if ((NX % PX) || (NY % PY) || (NZ % PZ)) {
        if (rank == 0) {
            fprintf(stderr, "Error: NX, NY, or NZ not divisible by PX, PY, or PZ.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    // Compute local subvolume dimensions.
    int nxLocal = NX / PX;
    int nyLocal = NY / PY;
    int nzLocal = NZ / PZ;

    // Process grid coordinates
    int px = rank % PX;                // Process X-coordinate
    int py = (rank / PX) % PY;         // Process Y-coordinate
    int pz = rank / (PX * PY);         // Process Z-coordinate

    // printf("Rank %d: Process grid coordinates: (%d, %d, %d)\n", rank, px, py, pz);
    // printf("Rank %d: Local subvolume dimensions: (%d, %d, %d)\n", rank, nxLocal, nyLocal, nzLocal);

    // Allocate local subvolume array for all components at once
    double** localData3D = (double**) malloc(NC * sizeof(double*));
    for (int c = 0; c < NC; c++) {
        localData3D[c] = (double*) malloc(nxLocal * nyLocal * nzLocal * sizeof(double));
        if (!localData3D[c]) {
            fprintf(stderr, "Rank %d: Error allocating localData3D for component %d\n", rank, c);
            MPI_Abort(MPI_COMM_WORLD, 4);
        }
    }

    double time1 = MPI_Wtime();

    // ---- Begin Parallel I/O ----

    MPI_File file;
    MPI_Offset disp = 0;

    MPI_Datatype file_type;
    int sizes[4]    = {NZ, NY, NX, NC};  // Global sizes: Z, Y, X, Components
    int subsizes[4] = {nzLocal, nyLocal, nxLocal, NC};  // Local block sizes
    int starts[4]   = {pz * nzLocal, py * nyLocal, px * nxLocal, 0};  // Where this rank starts

    // Create a subarray datatype
    MPI_Type_create_subarray(4, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &file_type);
    MPI_Type_commit(&file_type);

    // Open binary file for parallel reading
    if (MPI_File_open(MPI_COMM_WORLD, inputFile, MPI_MODE_RDONLY, MPI_INFO_NULL, &file) != MPI_SUCCESS) {
        if (rank == 0) fprintf(stderr, "Error: Cannot open binary input file %s\n", inputFile);
        MPI_Abort(MPI_COMM_WORLD, 99);
    }

    // Set view and read
    MPI_File_set_view(file, disp, MPI_DOUBLE, file_type, "native", MPI_INFO_NULL);

    // Allocate buffer for entire local block of all components
    double* readBuffer = (double*) malloc(nxLocal * nyLocal * nzLocal * NC * sizeof(double));
    if (!readBuffer) {
        fprintf(stderr, "Rank %d: Error allocating readBuffer\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 98);
    }

    // Perform parallel read
    MPI_File_read_all(file, readBuffer, nxLocal * nyLocal * nzLocal * NC, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&file);
    MPI_Type_free(&file_type);


    // Reorganize readBuffer into localData3D (per component)
    for (int c = 0; c < NC; c++) {
        for (long i = 0; i < (long)(nxLocal * nyLocal * nzLocal); i++) {
            localData3D[c][i] = readBuffer[i * NC + c];
        }
    }
    
    // printf("Rank %d, readBuffer:\n", rank);
    // for (int i = 0; i < nxLocal * nyLocal * nzLocal * NC; i++) {
    //     printf("%f ", readBuffer[i]);
    //     if ((i + 1) % (nxLocal * nyLocal * nzLocal) == 0) {
    //         printf("\n");
    //     }
    // }
    
    // // print local data for each process
    // for(int c = 0; c < NC; c++) {
    //     printf("Rank %d, Component %d:\n", rank, c);
    //     for (int z = 0; z < nzLocal; z++) {
    //         for (int y = 0; y < nyLocal; y++) {
    //             for (int x = 0; x < nxLocal; x++) {
    //                 printf("index: %d \n", IDX(x, y, z, nxLocal, nyLocal));
    //                 printf("%f ", localData3D[c][IDX(x, y, z, nxLocal, nyLocal)]);
    //             }
    //             printf("\n");
    //         }
    //         printf("\n");
    //     }
    // }
    free(readBuffer);

    // ---- End Parallel I/O ----

    double time2 = MPI_Wtime();  // I/O finished

    double ioTime = time2 - time1;
    double reducedIoTime;
    MPI_Reduce(&ioTime, &reducedIoTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // if (rank == 0) {
    //     printf("[DEBUG] I/O time: %lf seconds\n", time2 - time1);
    // }

    // Determine neighbors
    bool has_left_neighbour   = (px > 0);
    bool has_right_neighbour  = (px < PX - 1);
    bool has_front_neighbour  = (py > 0);
    bool has_back_neighbour   = (py < PY - 1);
    bool has_bottom_neighbour = (pz > 0);
    bool has_top_neighbour    = (pz < PZ - 1);
    
    int left_rank   = (has_left_neighbour)   ? rank - 1 : -1;
    int right_rank  = (has_right_neighbour)  ? rank + 1 : -1;
    int front_rank  = (has_front_neighbour)  ? rank - PX : -1;
    int back_rank   = (has_back_neighbour)   ? rank + PX : -1;
    int bottom_rank = (has_bottom_neighbour) ? rank - (PX * PY) : -1;
    int top_rank    = (has_top_neighbour)    ? rank + (PX * PY) : -1;

    // Arrays to store global min/max and local extrema counts
    long* localMinCount = (long*) calloc(NC, sizeof(long));
    long* localMaxCount = (long*) calloc(NC, sizeof(long));
    float* globalMinVal  = (float*) malloc(NC * sizeof(float));
    float* globalMaxVal  = (float*) malloc(NC * sizeof(float));
    for (int c = 0; c < NC; c++) {
        globalMinVal[c] =  1.0e30f;
        globalMaxVal[c] = -1.0e30f;
    }

    // Process all components in one go
    // double timeStartProcessing = MPI_Wtime();
    
    for (int c = 0; c < NC; c++) {
        // Allocate ghost cell buffers
        float *sendBottom = (float*) malloc(nxLocal * nyLocal * sizeof(float));
        float *recvBottom = (float*) malloc(nxLocal * nyLocal * sizeof(float));
        float *sendTop    = (float*) malloc(nxLocal * nyLocal * sizeof(float));
        float *recvTop    = (float*) malloc(nxLocal * nyLocal * sizeof(float));
        float *sendFront  = (float*) malloc(nxLocal * nzLocal * sizeof(float));
        float *recvFront  = (float*) malloc(nxLocal * nzLocal * sizeof(float));
        float *sendBack   = (float*) malloc(nxLocal * nzLocal * sizeof(float));
        float *recvBack   = (float*) malloc(nxLocal * nzLocal * sizeof(float));
        float *sendLeft   = (float*) malloc(nyLocal * nzLocal * sizeof(float));
        float *recvLeft   = (float*) malloc(nyLocal * nzLocal * sizeof(float));
        float *sendRight  = (float*) malloc(nyLocal * nzLocal * sizeof(float));
        float *recvRight  = (float*) malloc(nyLocal * nzLocal * sizeof(float));

        // Pack all ghost faces at once
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                // Bottom face (z == 0)
                for (int y = 0; y < nyLocal; y++) {
                    for (int x = 0; x < nxLocal; x++) {
                        sendBottom[y * nxLocal + x] = localData3D[c][IDX(x, y, 0, nxLocal, nyLocal)];
                    }
                }
            }
            
            #pragma omp section
            {
                // Top face (z == nzLocal - 1)
                for (int y = 0; y < nyLocal; y++) {
                    for (int x = 0; x < nxLocal; x++) {
                        sendTop[y * nxLocal + x] = localData3D[c][IDX(x, y, nzLocal - 1, nxLocal, nyLocal)];
                    }
                }
            }
            
            #pragma omp section
            {
                // Front face (y == 0)
                for (int z = 0; z < nzLocal; z++) {
                    for (int x = 0; x < nxLocal; x++) {
                        sendFront[z * nxLocal + x] = localData3D[c][IDX(x, 0, z, nxLocal, nyLocal)];
                    }
                }
            }
            
            #pragma omp section
            {
                // Back face (y == nyLocal - 1)
                for (int z = 0; z < nzLocal; z++) {
                    for (int x = 0; x < nxLocal; x++) {
                        sendBack[z * nxLocal + x] = localData3D[c][IDX(x, nyLocal - 1, z, nxLocal, nyLocal)];
                    }
                }
            }
            
            #pragma omp section
            {
                // Left face (x == 0)
                for (int z = 0; z < nzLocal; z++) {
                    for (int y = 0; y < nyLocal; y++) {
                        sendLeft[z * nyLocal + y] = localData3D[c][IDX(0, y, z, nxLocal, nyLocal)];
                    }
                }
            }
            
            #pragma omp section
            {
                // Right face (x == nxLocal - 1)
                for (int z = 0; z < nzLocal; z++) {
                    for (int y = 0; y < nyLocal; y++) {
                        sendRight[z * nyLocal + y] = localData3D[c][IDX(nxLocal - 1, y, z, nxLocal, nyLocal)];
                    }
                }
            }
        }

        // Set up all communication at once
        MPI_Request requests[12];  // For up to 12 operations (6 sends + 6 receives)
        MPI_Status statuses[12];
        int req_count = 0;

        // Post all receives first (non-blocking)
        if (has_bottom_neighbour) {
            MPI_Irecv(recvBottom, nxLocal * nyLocal, MPI_FLOAT, bottom_rank, 0, MPI_COMM_WORLD, &requests[req_count++]);
        }
        if (has_top_neighbour) {
            MPI_Irecv(recvTop, nxLocal * nyLocal, MPI_FLOAT, top_rank, 1, MPI_COMM_WORLD, &requests[req_count++]);
        }
        if (has_front_neighbour) {
            MPI_Irecv(recvFront, nxLocal * nzLocal, MPI_FLOAT, front_rank, 2, MPI_COMM_WORLD, &requests[req_count++]);
        }
        if (has_back_neighbour) {
            MPI_Irecv(recvBack, nxLocal * nzLocal, MPI_FLOAT, back_rank, 3, MPI_COMM_WORLD, &requests[req_count++]);
        }
        if (has_left_neighbour) {
            MPI_Irecv(recvLeft, nyLocal * nzLocal, MPI_FLOAT, left_rank, 4, MPI_COMM_WORLD, &requests[req_count++]);
        }
        if (has_right_neighbour) {
            MPI_Irecv(recvRight, nyLocal * nzLocal, MPI_FLOAT, right_rank, 5, MPI_COMM_WORLD, &requests[req_count++]);
        }

        // Then post all sends (non-blocking)
        if (has_bottom_neighbour) {
            MPI_Isend(sendBottom, nxLocal * nyLocal, MPI_FLOAT, bottom_rank, 1, MPI_COMM_WORLD, &requests[req_count++]);
        }
        if (has_top_neighbour) {
            MPI_Isend(sendTop, nxLocal * nyLocal, MPI_FLOAT, top_rank, 0, MPI_COMM_WORLD, &requests[req_count++]);
        }
        if (has_front_neighbour) {
            MPI_Isend(sendFront, nxLocal * nzLocal, MPI_FLOAT, front_rank, 3, MPI_COMM_WORLD, &requests[req_count++]);
        }
        if (has_back_neighbour) {
            MPI_Isend(sendBack, nxLocal * nzLocal, MPI_FLOAT, back_rank, 2, MPI_COMM_WORLD, &requests[req_count++]);
        }
        if (has_left_neighbour) {
            MPI_Isend(sendLeft, nyLocal * nzLocal, MPI_FLOAT, left_rank, 5, MPI_COMM_WORLD, &requests[req_count++]);
        }
        if (has_right_neighbour) {
            MPI_Isend(sendRight, nyLocal * nzLocal, MPI_FLOAT, right_rank, 4, MPI_COMM_WORLD, &requests[req_count++]);
        }

        // While waiting for communication to complete, we can compute internal points
        long thisMinCount = 0;
        long thisMaxCount = 0;
        float thisLocalMin = 1.0e30f;
        float thisLocalMax = -1.0e30f;

        // Find global min/max first (doesn't need neighbor data)
        for (int z = 0; z < nzLocal; z++) {
            for (int y = 0; y < nyLocal; y++) {
                for (int x = 0; x < nxLocal; x++) {
                    float val = localData3D[c][IDX(x, y, z, nxLocal, nyLocal)];
                    if (val < thisLocalMin) thisLocalMin = val;
                    if (val > thisLocalMax) thisLocalMax = val;
                }
            }
        }

        // Wait for all MPI communications to complete
        MPI_Waitall(req_count, requests, statuses);

        // Now check for local extrema including ghost cells
        for (int z = 0; z < nzLocal; z++) {
            for (int y = 0; y < nyLocal; y++) {
                for (int x = 0; x < nxLocal; x++) {
                    float val = localData3D[c][IDX(x, y, z, nxLocal, nyLocal)];
                    
                    int isLocalMin = 1;
                    int isLocalMax = 1;
                    int neighCount = 0;
                    
                    // Check all six neighbors (where they exist)
                    
                    // Left neighbor
                    if (x > 0) {
                        float nval = localData3D[c][IDX(x - 1, y, z, nxLocal, nyLocal)];
                        if (val >= nval) isLocalMin = 0;
                        if (val <= nval) isLocalMax = 0;
                        neighCount++;
                    } else if (has_left_neighbour) {
                        float nval = recvLeft[z * nyLocal + y];
                        if (val >= nval) isLocalMin = 0;
                        if (val <= nval) isLocalMax = 0;
                        neighCount++;
                    }
                    
                    // Right neighbor
                    if (x < nxLocal - 1) {
                        float nval = localData3D[c][IDX(x + 1, y, z, nxLocal, nyLocal)];
                        if (val >= nval) isLocalMin = 0;
                        if (val <= nval) isLocalMax = 0;
                        neighCount++;
                    } else if (has_right_neighbour) {
                        float nval = recvRight[z * nyLocal + y];
                        if (val >= nval) isLocalMin = 0;
                        if (val <= nval) isLocalMax = 0;
                        neighCount++;
                    }
                    
                    // Front neighbor
                    if (y > 0) {
                        float nval = localData3D[c][IDX(x, y - 1, z, nxLocal, nyLocal)];
                        if (val >= nval) isLocalMin = 0;
                        if (val <= nval) isLocalMax = 0;
                        neighCount++;
                    } else if (has_front_neighbour) {
                        float nval = recvFront[z * nxLocal + x];
                        if (val >= nval) isLocalMin = 0;
                        if (val <= nval) isLocalMax = 0;
                        neighCount++;
                    }
                    
                    // Back neighbor
                    if (y < nyLocal - 1) {
                        float nval = localData3D[c][IDX(x, y + 1, z, nxLocal, nyLocal)];
                        if (val >= nval) isLocalMin = 0;
                        if (val <= nval) isLocalMax = 0;
                        neighCount++;
                    } else if (has_back_neighbour) {
                        float nval = recvBack[z * nxLocal + x];
                        if (val >= nval) isLocalMin = 0;
                        if (val <= nval) isLocalMax = 0;
                        neighCount++;
                    }
                    
                    // Bottom neighbor
                    if (z > 0) {
                        float nval = localData3D[c][IDX(x, y, z - 1, nxLocal, nyLocal)];
                        if (val >= nval) isLocalMin = 0;
                        if (val <= nval) isLocalMax = 0;
                        neighCount++;
                    } else if (has_bottom_neighbour) {
                        float nval = recvBottom[y * nxLocal + x];
                        if (val >= nval) isLocalMin = 0;
                        if (val <= nval) isLocalMax = 0;
                        neighCount++;
                    }
                    
                    // Top neighbor
                    if (z < nzLocal - 1) {
                        float nval = localData3D[c][IDX(x, y, z + 1, nxLocal, nyLocal)];
                        if (val >= nval) isLocalMin = 0;
                        if (val <= nval) isLocalMax = 0;
                        neighCount++;
                    } else if (has_top_neighbour) {
                        float nval = recvTop[y * nxLocal + x];
                        if (val >= nval) isLocalMin = 0;
                        if (val <= nval) isLocalMax = 0;
                        neighCount++;
                    }
                    
                    // Only count as extrema if we have at least one neighbor to compare with
                    if (neighCount > 0) {
                        if (isLocalMin) thisMinCount++;
                        if (isLocalMax) thisMaxCount++;
                    }
                }
            }
        }

        // Reduction operations for this component
        MPI_Reduce(&thisLocalMin, &globalMinVal[c], 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&thisLocalMax, &globalMaxVal[c], 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&thisMinCount, &localMinCount[c], 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&thisMaxCount, &localMaxCount[c], 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        // Free ghost cell buffers
        free(sendBottom); free(recvBottom);
        free(sendTop);    free(recvTop);
        free(sendFront);  free(recvFront);
        free(sendBack);   free(recvBack);
        free(sendLeft);   free(recvLeft);
        free(sendRight);  free(recvRight);
    }

    double time3 = MPI_Wtime();
    double mainCodeTime = time3 - time2;
    double reducedMainCodeTime;
    MPI_Reduce(&mainCodeTime, &reducedMainCodeTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    // if (rank == 0) {
    //     printf("[DEBUG] Processing time: %lf seconds\n", time3 - time2);
    // }

    // Write output results
    FILE* fout;
    if (rank == 0) {
        fout = fopen(outputFile, "w");
        if (!fout) {
            fprintf(stderr, "Cannot open output file %s\n", outputFile);
            MPI_Abort(MPI_COMM_WORLD, 8);
        }
        // fprintf(fout, "Rank %d: Min/Max counts and values for each component:\n", rank);
        // fprintf(fout, "Min/Max counts:\n");
        for (int c = 0; c < NC; c++) {
            fprintf(fout, "(%ld,%ld)", localMinCount[c], localMaxCount[c]);
            if (c < NC - 1) fprintf(fout, ",");
        }
        fprintf(fout, "\n");
        for (int c = 0; c < NC; c++) {
            fprintf(fout, "(%.2f,%.2f)", globalMinVal[c], globalMaxVal[c]);
            if (c < NC - 1) fprintf(fout, ",");
        }
        fprintf(fout, "\n");
    }
    double time4 = MPI_Wtime();
    double totalTime = time4 - time1;
    double reducedTotalTime;
    MPI_Reduce(&totalTime, &reducedTotalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if(rank == 0) {
        fprintf(fout, "%lf, %lf, %lf", reducedIoTime, reducedMainCodeTime, reducedTotalTime);
        fprintf(fout, "\n");
        fclose(fout);
    }
    
    // Free allocated memory
    for (int c = 0; c < NC; c++) {
        free(localData3D[c]);
    }
    free(localData3D);
    free(localMinCount);
    free(localMaxCount);
    free(globalMinVal);
    free(globalMaxVal);

    // if (rank == 0) {
    //     printf("[DEBUG] Total runtime: %lf seconds\n", totalTime);
    // }
    
    MPI_Finalize();
    return 0;
}