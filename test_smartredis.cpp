#include "client.h" // from smartredis
#include <vector>

using namespace std;

void testSmartRedis() {
    std::cout << "Testing SmartRedis..." << std::endl;
    
    size_t dim1 = 3;
    size_t dim2 = 2;
    size_t dim3 = 5;
    vector<size_t> dims = {dim1, dim2, dim3};
    size_t nvalues = dim1 * dim2 * dim3;
    vector<double> input_tensor(nvalues, 0.0);
    for(size_t i=0; i<nvalues; i++){
	    input_tensor[i] = 2.0*rand()/(double)RAND_MAX - 1.0;
    }

    // Initialize a SamrtRedis client 
    SmartRedis::Client client("client1");
    
    // Put the tensor in the database
    string key = "3d_tensor";
    client.put_tensor(key, input_tensor.data(), dims, SRTensorTypeDouble, SRMemLayoutContiguous);

    // Retrieve the tensor from the database using the unpack feature
    vector<double> unpack_tensor(nvalues, 0.0);
    client.unpack_tensor(key, unpack_tensor.data(), {nvalues}, SRTensorTypeDouble, SRMemLayoutContiguous);

    // Print the values retrieved with the unpack feature
    for(size_t i=0; i<nvalues; i++){
        cout << "Sent: " << input_tensor[i] << endl;
	cout << "received: " << unpack_tensor[i] << endl;
    };
    
    // Retrieve the tensor from the database with the get feature
    SRTensorType get_type;
    vector<size_t> get_dims;
    void* get_tensor;
    client.get_tensor(key, get_tensor, get_dims, get_type, SRMemLayoutNested);

    // Print retrieved tensor with the get feature
    for(size_t i=0, c=0; i<dims[0]; i++)
        for(size_t j=0; j<dims[1]; j++)
            for(size_t k=0; k<dims[2]; k++, c++) {
                std::cout<<"Sent: "<<input_tensor[c]<<" "
                         <<"Received: "
                         <<((double***)get_tensor)[i][j][k]<<std::endl;

    }
    cout << "SmartRedis test completed." << endl;
}

int main() {
    testSmartRedis();
    return 0;
}

