#include "parasymbolic.hpp"

SparseMatrix::SparseMatrix(size_t size, MagicOperator* op):
size(size), op(op), _nz_count(0){
    row_lens = new size_t[size];
    row_indices = new size_t*[size]
    row_probs = new dtype*[size];
    row_values = new dtype*[size];

    prob_row_coefficients = new dtype[size];
    prob_col_coefficients = new dtype[size];
    value_row_offsets = new dtype[size];
    value_column_offsets = new dtype[size];
    for (size_t i = 0; i < size; i++){
        row_indices[i] = NULL;
        prob_row_coefficients[i] = prob_col_coefficients[i] = 1;
        value_row_offsets[i] = value_column_offsets[i] = 0;
    }
}

void SparseMatrix::batch_set(size_t row, size_t const* A_columns, size_t c_len, dtype const* A_probs, size_t p_len, dtype const* A_values, size_t v_len){
    _nz_count += c_len;

    row_lens[row] = c_len;
    row_indices[row] = new size_t[c_len];
    row_probs[row] = new dtype[c_len];
    row_values[row] = new dtype[c_len];

    for(size_t c = 0; c < c_len; c++){
        row_indices[row][c] = A_columns[c];
        row_probs[row][c] = A_probs[c];
        row_values[row][c] = A_values[c];
    }
}

int binary_search(size_t* haystack, size_t len, size_t needle){
    if (len == 0)
        return -1;
    int start = 0;
    int end = len-1;
    while (start <= end){
        auto mid = (start+end)/2;
        auto v = haystack[mid];
        if (v == needle)
            return mid;
        if (v < needle)
            start = mid+1;
        else
            end = mid-1;
    }
    return -1;
}

bool SparseMatrix::has_value(size_t row, size_t column){
    int ind = binary_search(row_indices[row], row_lens[row], column)
    return (ind != -1);
}

std::tuple<dtype, dtype> SparseMatrix::get(size_t row,size_t column){
    int ind = binary_search(row_indices[row], row_lens[row], column);
    return std::make_tuple(row_probs[row][ind], row_values[row][ind]);
}

size_t SparseMatrix::nz_count(){
    return _nz_count;
}

ManifestMatrix manifest(dtype const* A_rolls, size_t r_len){
    ManifestMatrix ret(this);
    size_t roll_ind = 0;
    for (size_t r = 0; r < origin->size; r++){
        auto len = origin->row_lens[r];
        auto row_mul = origin->prob_row_coefficients[r];
        auto row_prob = origin->row_probs[r];
        auto im = ret.is_manifest[r];
        for (size_t c = 0; c < len; c++){
            auto c_i = row_indices[c];
            auto effective_m = row_mul*(origin->prob_col_coefficients[c_i])*row_prob[c];
            im[c] = A_rolls[roll_ind++] < effective_m;
        }
    }
    return ret;
}

SparseMatrix::~SparseMatrix(){
    for (size_t i = 0; i < size; i++){
        if (row_indices[i]){
            delete[] row_indices[i];
            delete[] row_probs[i];
            delete[] row_values[i];
        }
    }
    delete[] row_lens;
    delete[] row_indices;
    delete[] row_probs;
    delete[] row_values;

    delete[] prob_row_coefficients;
    delete[] prob_col_coefficients;
    delete[] value_row_offsets;
    delete[] value_column_offsets;
}

ManifestMatrix::ManifestMatrix(SparseMatrix* origin):
    origin(origin){
    is_manifest = new bool*[origin->size];
    for (size_t r = 0; r < origin->size; r++){
        is_manifest[r] = new bool[origin->row_lens[r]];
    }
}