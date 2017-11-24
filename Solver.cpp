#include "Solver.h"
#include <string>

using namespace std;

/// Definitions for the CUDA Kernel calls and Kernels
extern "C"
{
	/// Calculates the grid and block sizes
	void computeGridSize(unsigned int n, unsigned int blockSize, unsigned int &numBlocks, unsigned int &numThreads);

	/// Calls a function and checks if an error happened
	void HandleError(cudaError_t status, string message);

	/// Copies the simulation parameters to constant memory
	void SetParameters(Parameters *p);

	/// Calculats the hash of every particle: calculates the cell where it is in the grid
	void CalHash(unsigned int* index, unsigned int* hash, float3* pos, unsigned int num_particles);

	/// Sorts the particles according to their hash
	void SortParticles(unsigned int *hash, unsigned int *index, unsigned int num_particles);

	/// Reorders the particles according to the sorted hash
	void ReorderDataAndFindCellStart(unsigned int* cellstart,
		unsigned int* cellend,
		float3* spos,
		float3* svel,
		unsigned int* hash,
		unsigned int* index,
		float3* pos,
		float3* vel,
		unsigned int num_particles,
		unsigned int gridNum);

	void CalcDensityPressure(float* dens,
					float* pres,
		             unsigned int* cellstart, 
					 unsigned int* cellend, 
					 float3 *spos, 
					 unsigned int num_particles);

	void CalcForce(float3* force, 
		           float3* spos, 
				   float3* svel, 
				   float3* vel, 
				   float* press, 
				   float* dens, 
				   unsigned int* index, 
				   unsigned int* cellstart, 
				   unsigned int* cellend, 
				   unsigned int num_particles);

	void UpdateVelocityAndPosition(float3* pos, 
		                           float3* vel, 
								   float3* force, 
								   unsigned int num_particles);

	void HandleBoundary(float3* pos, 
		                float3* vel, 
						unsigned int num_particles);

}

#define CHECK(ptr, message)  {if(ptr==NULL){cerr<<message<<endl;exit(1);}}

Solver::Solver(unsigned int _num_particles) :num_particles(_num_particles)
{
	size1 = num_particles*sizeof(float);
	size3 = num_particles*sizeof(float3);
	gridNum = GRID_SIZE * GRID_SIZE * GRID_SIZE;

	/// Set simulation parameters
	pa.mass = 0.5f;
	pa.dt = 0.005f;

	pa.xmin = 0.0f;
	pa.xmax = GRID_SIZE;
	pa.ymin = 0.0f;
	pa.ymax = GRID_SIZE;
	pa.zmin = 0.0f;
	pa.zmax = GRID_SIZE;

	pa.gridSize.x = GRID_SIZE;
	pa.gridSize.y = GRID_SIZE;
	pa.gridSize.z = GRID_SIZE;
	pa.cellSize = 1;

	pa.h = 1.5f;
	pa.k = 10.0f;
	pa.restDens = 0.2f;
	pa.mu = 0.5f;

	/// Memory allocation
	hpos=(float3*)malloc(size3);
	CHECK(hpos, "Failed to allocate memory of hpos!");

	hvel = (float3*)malloc(size3);
	CHECK(hvel, "Failed to allocate memory of hvel!");

	HandleError(cudaMalloc((void**) &dpos, size3), "Failed to allocate memory of dpos!");
	HandleError(cudaMalloc((void**) &dvel, size3), "Failed to allocate memory of dvel!");
	HandleError(cudaMalloc((void**) &dspos, size3), "Failed to allocate memory of dspos!");
	HandleError(cudaMalloc((void**) &dsvel, size3), "Failed to allocate memory of dsvel!");
	HandleError(cudaMalloc((void**) &ddens, size1), "Failed to allocate memory of ddens!");
	HandleError(cudaMalloc((void**) &dforce, size3), "Failed to allocate memory of dforce!");
	HandleError(cudaMalloc((void**) &dpress, size1), "Failed to allocate memory of dpress!");

	HandleError(cudaMalloc((void**) &dindex, num_particles*sizeof(unsigned int)), "Failed to allocate memory of dindex");	
	HandleError(cudaMalloc((void**) &dhash, num_particles*sizeof(unsigned int)), "Failed to allocate memory of dhash");
	HandleError(cudaMalloc((void**) &dcellStart, gridNum*sizeof(unsigned int)), "Failed to allocate memory of dcellstart");
	HandleError(cudaMalloc((void**) &dcellEnd, gridNum*sizeof(unsigned int)), "Failed to allocate memory of dcellend");

	InitParticles();

	HandleError(cudaMemcpy(dpos, hpos, size3, cudaMemcpyHostToDevice), "Failed to copy memory of hpos!");
	HandleError(cudaMemset(dvel, 0, size3), "Failed to memset dvel!");
	HandleError(cudaMemset(dsvel, 0, size3), "Failed to memset dsvel!"); 
	HandleError(cudaMemset(dspos, 0, size3), "Failed to memset dspos!"); 
	HandleError(cudaMemset(ddens, 0, size1), "Failed to memset ddens!");
	HandleError(cudaMemset(dforce, 0, size3), "Failed to memset dforce!");
	HandleError(cudaMemset(dpress, 0, size1), "Failed to memset dpress!");

	HandleError(cudaMemset(dindex, 0, size1), "Failed to memset dindex!");
	HandleError(cudaMemset(dhash, 0, size1), "Failed to memset dhash!");
	HandleError(cudaMemset(dcellStart, 0, gridNum*sizeof(unsigned int)), "Failed to memset dcellstart!");
	HandleError(cudaMemset(dcellEnd, 0, gridNum*sizeof(unsigned int)), "Failed to memset dcellend!");

	SetParameters(&pa);
}

Solver::~Solver()
{
	free(hpos);
	free(hvel);

	HandleError(cudaFree(dpos), "Failed to free dpos!");
	HandleError(cudaFree(dvel), "Failed to free dvel!");
	HandleError(cudaFree(ddens), "Failed to free ddens!");
	HandleError(cudaFree(dforce), "Failed to free dforce!");
	HandleError(cudaFree(dpress), "Failed to free dpress!");
	HandleError(cudaFree(dhash), "Failed to free dhash!");
	HandleError(cudaFree(dindex), "Failed to free dindex!");
	HandleError(cudaFree(dcellStart), "Failed to free dcellStart!");
	HandleError(cudaFree(dcellEnd), "Failed to free dcellEnd!");

	HandleError(cudaFree(dspos), "Failed to free dspos!");
	HandleError(cudaFree(dsvel), "Failed to free dsvel!");
}


void Solver::InitParticles()
{
	/// Initializing a set number of particles
	int index = 0;
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			for (int k = 0; k < 16; k++)
			{
				index = k * 32 * 32 + j * 32 + i;
				hpos[index].x = i;
				hpos[index].y = j;
				hpos[index].z = k;
			}
		}
	}
}

void Solver::Update()
{
	CalHash(dindex, dhash, dpos, num_particles);

	SortParticles(dhash, dindex, num_particles);

	ReorderDataAndFindCellStart(dcellStart, dcellEnd, dspos, dsvel, dhash, dindex, dpos, dvel, num_particles, gridNum);
	
	CalcDensityPressure(ddens, dpress, dcellStart, dcellEnd, dspos, num_particles);

	CalcForce(dforce, dspos, dsvel, dvel, dpress, ddens, dindex, dcellStart, dcellEnd, num_particles);

	UpdateVelocityAndPosition(dpos, dvel, dforce, num_particles);

	HandleBoundary(dpos, dvel, num_particles);

	HandleError(cudaMemcpy(hpos, dpos, size3, cudaMemcpyDeviceToHost), "Failed to copy device to host in update!");
}
