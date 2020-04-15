#include "sparse.hpp"
#include <iostream>

dtype MagicOperator::operate(dtype w_val, dtype v_val) const{
    return 1 - w_val*v_val;
}

MagicOperator* MagicOperator::mul(dtype factor) const{
    return new FactoredMagicOperator(this, factor);
}

FactoredMagicOperator::FactoredMagicOperator(MagicOperator const* inner, dtype factor): inner(inner), factor(factor){}
dtype FactoredMagicOperator::operate(dtype w_val, dtype v_val) const{
    return inner->operate(w_val, v_val) * factor;
}
MagicOperator* FactoredMagicOperator::mul(dtype factor) const{
    return inner->mul(factor * this->factor);
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

int SparseMatrix::col_ind(size_t row, size_t column) const{
    return binary_search(row_indices[row], row_lens[row], column);
}

SparseMatrix::SparseMatrix(size_t size):
size(size), _nz_count(0){
    row_lens = new size_t[size];
    row_indices = new size_t*[size];
    row_probs = new dtype*[size];
    row_values = new dtype*[size];

    prob_row_coefficients = new dtype[size];
    prob_column_coefficients = new dtype[size];
    value_row_offsets = new dtype[size];
    value_column_offsets = new dtype[size];
    for (size_t i = 0; i < size; i++){
        row_indices[i] = NULL;
        row_lens[i] = 0;

        prob_row_coefficients[i] = prob_column_coefficients[i] = 1;
        value_row_offsets[i] = value_column_offsets[i] = 0;
    }
}

void SparseMatrix::batch_set(size_t row, size_t const* A_columns, size_t c_len, dtype const* A_probs, size_t p_len, dtype const* A_values, size_t v_len){
    _nz_count += c_len;

    if (row_indices[row]){
        std::cerr << "row " << row <<" already assigned" <<std::endl;
        throw "";
    }

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

bool SparseMatrix::has_value(size_t row, size_t column) const{
    return col_ind(row,column) != -1;
}

std::tuple<dtype, dtype> SparseMatrix::get(size_t row,size_t column) const{
    int ind = col_ind(row,column);
    return std::make_tuple(row_probs[row][ind], row_values[row][ind]);
}

size_t SparseMatrix::nz_count(){
    return _nz_count;
}

ManifestMatrix* SparseMatrix::manifest(dtype const* A_rolls, size_t r_len){
    auto ret = new ManifestMatrix(this);
    size_t roll_ind = 0;
    for (size_t r = 0; r < size; r++){
        auto len = row_lens[r];
        auto row_ind = row_indices[r];
        auto row_mul = prob_row_coefficients[r];
        auto row_prob = row_probs[r];
        auto im = ret->is_manifest[r];
        for (size_t c = 0; c < len; c++){
            auto c_i = row_ind[c];
            auto effective_m = row_prob[c] * row_mul * prob_column_coefficients[c_i];
            im[c] = A_rolls[roll_ind++] < effective_m;
        }
    }
    return ret;
}

void SparseMatrix::row_set_prob_coff(size_t row, dtype coff){
    prob_row_coefficients[row] = coff;
}
void SparseMatrix::col_set_prob_coff(size_t column, dtype coff){
    prob_column_coefficients[column] = coff;
}

void SparseMatrix::row_set_value_offset(size_t row, dtype offset){
    value_row_offsets[row] = offset;
}
void SparseMatrix::col_set_value_offset(size_t column, dtype offset){
    value_column_offsets[column] = offset;
}

std::vector<std::vector<size_t>> SparseMatrix::non_zero_columns(){
    std::vector<std::vector<size_t>> ret(size);
    for (int i = 0; i < size; i++){
        auto len = row_lens[i];
        auto arr = row_indices[i];
        for (auto j = 0; j < len; j++){
            ret[i].push_back(arr[j]);
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
    delete[] prob_column_coefficients;
    delete[] value_row_offsets;
    delete[] value_column_offsets;
}

ManifestMatrix::ManifestMatrix(SparseMatrix* const & origin):
    origin(origin){
    is_manifest = new bool*[origin->size];
    for (size_t r = 0; r < origin->size; r++){
        is_manifest[r] = new bool[origin->row_lens[r]];
    }
}

dtype ManifestMatrix::get(size_t row,size_t column){
    auto c_i = origin->col_ind(row, column);
    if ((c_i != -1) && this->is_manifest[row][c_i]){
        return std::get<1>(origin->get(row, column)) + origin->value_row_offsets[row] + origin->value_column_offsets[column];
    }
    return 0;
}

void ManifestMatrix::I_POA(dtype const* A_values, size_t v_len, size_t const* A_nz_indices, size_t nzi_len, MagicOperator* const& op, dtype** AF_out, size_t* o_len){
    *AF_out = new dtype[origin->size];
    *o_len = origin->size;
    for (size_t row = 0; row < origin->size; row++){
        dtype total = 1;

        size_t nzi = 0;
        size_t column_index = 0;
        auto row_len = origin->row_lens[row];
        auto row_offset = origin->value_row_offsets[row];
        auto row_indices = origin->row_indices[row];
        auto row_values = origin->row_values[row];
        auto manifest_row = is_manifest[row];

        while (nzi < nzi_len && column_index < row_len){
            if (!manifest_row[column_index]){
                column_index++;
                continue;
            }
            auto r_column = row_indices[column_index];
            auto v_column = A_nz_indices[nzi];
            if (r_column < v_column){
                column_index++;
            }
            else if (v_column < r_column)
            {
                nzi++;
            }
            else{
                auto r_val = row_values[column_index] + row_offset + origin->value_column_offsets[r_column];
                auto v_val = A_values[v_column];
                auto m = op->operate(r_val, v_val);
                total *= m;
                column_index++;
                nzi++;
            }
        }

        (*AF_out)[row] = total;
    }
}

std::vector<std::vector<size_t>> ManifestMatrix::nz_rows(){
    std::vector<std::vector<size_t>> ret;
    for (size_t row = 0; row < origin->size; row++){
        std::vector<size_t> row_val;
        auto len = origin->row_lens[row];
        auto columns = origin->row_indices[row];
        auto im = is_manifest[row];
        for (size_t c = 0; c < len; c++){
            if (im[c])
                row_val.push_back(columns[c]);
        }
        ret.push_back(row_val);
    }
    return ret;
}

ManifestMatrix::~ManifestMatrix(){
    for (size_t r = 0; r < origin->size; r++){
        delete[] is_manifest[r];
    }
    delete[] is_manifest;
}

WMaskedTabularOp::WMaskedTabularOp(dtype v_res, size_t v_len, size_t const* A_mask_len, size_t ml_len,
                                                               dtype const * A_table, size_t table_len):
    v_res(v_res), v_len(v_len){
    memcpy(mask_len, A_mask_len, ml_len*sizeof(size_t));
    for (auto i = ml_len; i < mask_count; i++){
        mask_len[i] = 1;
    }
    table = new dtype[table_len];
    memcpy(table, A_table, table_len*sizeof(dtype));

    increments[0] = v_len;
    for (auto i = 1; i < mask_count; i++){
        increments[i] = increments[i-1] * mask_len[i-1];
    }
}
dtype WMaskedTabularOp::operate(dtype w_val, dtype v_val) const{
    auto w_chars = reinterpret_cast<unsigned char *>(&w_val);
    size_t index = (int)(v_val / v_res);
    for (auto i = 0; i < mask_count; i++){
        index += increments[i] * w_chars[i];
    }
    return table[index];
}
WMaskedTabularOp::~WMaskedTabularOp(){
    delete[] table;
}

dtype from_bytes(unsigned char const* A_bytes, size_t b_len){
    unsigned char bytes[mask_count];
    memcpy(bytes, A_bytes, b_len);
    for (auto i = b_len; i < mask_count; i++){
        bytes[i] = 0;
    }
    auto ref = reinterpret_cast<float const *>(bytes);
    return *ref;
}