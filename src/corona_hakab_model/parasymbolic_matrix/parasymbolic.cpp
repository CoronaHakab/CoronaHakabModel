#include "parasymbolic.hpp"
#include <math.h>
#include <iostream>
#include <thread>
#include <algorithm>

#define POOL_SIZE 2

// region bare
//todo merge bare and coffs
//todo function to get pool_size
BareSparseMatrix::BareSparseMatrix(size_t size): size(size), total(0){
    rows = new row_type[size];
    columns = new col_type[size];
}

void BareSparseMatrix::set(size_t r, size_t c, dtype v){
    // we don't even bother changing the column set since this is never called to set a non-zero index
    auto row = rows[r];
    // and to prove it...
    auto prev = row[c];
    rows[r][c] = v;
    total += v - prev;
}

dtype BareSparseMatrix::get(size_t row_ind, size_t column_ind){
    auto row = rows[row_ind];
    auto found = row.find(column_ind);
    if (found == row.end())
        return 0;
    return found->second;
}

void BareSparseMatrix::batch_set(size_t row_num,
                                 size_t const * columns, size_t c_len,
                                 dtype const* values, size_t v_len){
    auto& row = rows[row_num];
    auto iter = row.begin();
    auto tup_index = 0;
    while (true){
        auto next_i_col = columns[tup_index];
        auto next_i_val = values[tup_index];
        if ((next_i_val == 0) && ++tup_index == c_len) return;

        while (iter != row.end() && iter->first < next_i_col){
            if (++iter == row.end()) break;
        }
        if (iter == row.end()){
            for (auto i = tup_index; i < c_len; ++i){
                auto res = row.insert({columns[i], values[i]});
                this->columns[columns[i]].insert(row_num);
                this->total += values[i];
            }
            return;
        }
        if (iter->first == next_i_col){
            this->total += next_i_val - iter->second;
            iter->second = next_i_val;
            if (++tup_index == c_len) return;
        }
        else { // iter->first > next_i_col
            iter = row.insert(iter, {next_i_col, next_i_val});
            this->columns[columns[next_i_col]].insert(row_num);
            this->total += next_i_val;
            if (++tup_index == c_len) return;
        }
    }
}

double BareSparseMatrix::get_total(){
    if (!isnan(total)){
        return total;
    }
    // kahan sum
    double sum = 0;
    double c = 0;
    for (auto row_ind = 0; row_ind < size; row_ind++){
        auto& row = rows[row_ind];
        for (auto pair: row){
            auto y = pair.second - c;
            auto t = sum + y;
            c = (t - sum)  - y;
            sum = t;
        }
    }
    return sum;
}

BareSparseMatrix::~BareSparseMatrix(){
    delete[] rows;
    delete[] columns;
}
// endregion
// region CoffedSparseMatrix
CoffedSparseMatrix::CoffedSparseMatrix(size_t size) : BareSparseMatrix(size){
    col_coefficients = new dtype[size];
    row_coefficients = new dtype[size];
    for (auto i = 0; i < size; i++){
        col_coefficients[i] = row_coefficients[i] = 1.0;
    }
}

dtype CoffedSparseMatrix::get(size_t row, size_t column){
    auto coffs = row_coefficients[row] * col_coefficients[column];
    if (coffs == 0)
        return 0;
    return BareSparseMatrix::get(row, column) * coffs;
}

void CoffedSparseMatrix::mul_row(size_t row, dtype factor){
    row_coefficients[row] *= factor;
    total = NAN;
}

void CoffedSparseMatrix::mul_col(size_t col, dtype factor){
    col_coefficients[col] *= factor;
    total = NAN;
}

void CoffedSparseMatrix::reset_mul_row(size_t row){
    row_coefficients[row] = 1.0;
    total = NAN;
}

void CoffedSparseMatrix::reset_mul_col(size_t col){
    col_coefficients[col] = 1.0;
    total = NAN;
}

CoffedSparseMatrix::~CoffedSparseMatrix(){
    delete[] row_coefficients;
    delete[] col_coefficients;
}
// endregion
// region parasymbolic
FastSparseMatrix::FastSparseMatrix(size_t size): size(size){
    indices = new std::vector<size_t>[size];
    data = new std::vector<dtype>[size];
    columns = new col_type[size];
}
FastSparseMatrix::~FastSparseMatrix(){
    delete[] indices;
    delete[] data;
    delete[] columns;
}

ParasymbolicMatrix::ParasymbolicMatrix(size_t size, size_t component_count):
 component_count(component_count), inner(size), calc_lock(false){
    factors = new dtype[component_count];
    components = new CoffedSparseMatrix*[component_count];
    for (auto i = 0; i < component_count; i++){
        factors[i] = 1;
        components[i] = new CoffedSparseMatrix(size);
    }
    pool = new ctpl::thread_pool(POOL_SIZE);
    rebuild_all();
}

void ParasymbolicMatrix::rebuild_all(){
    // todo parallelize
    for(auto row_num = 0; row_num < inner.size; row_num++){
        rebuild_row(row_num);
    }
}
void ParasymbolicMatrix::rebuild_row(size_t row_num){
    // todo most of the time, we'll only need to reset some cells in each row (only non-zero in the changed cells)
    auto& row_indices = inner.indices[row_num];
    auto& row_data = inner.data[row_num];

    row_indices.clear();
    row_data.clear();

    using row_iter = decltype(components[0]->rows[0].cbegin());

    row_iter* comp_iters = new row_iter[component_count];
    row_iter* comp_ends = new row_iter[component_count];
    dtype** comp_col_coffs = new dtype*[component_count];
    dtype* comp_row_coffs = new dtype[component_count];
    for (auto comp_num = 0; comp_num < component_count; comp_num++){
        comp_ends[comp_num] = components[comp_num]->rows[row_num].cend();
        comp_row_coffs[comp_num] = components[comp_num]->row_coefficients[row_num];
        comp_iters[comp_num] = components[comp_num]->rows[row_num].cbegin();
        comp_col_coffs[comp_num] = components[comp_num]->col_coefficients;
    }
    auto min_set = std::unordered_set<size_t>();
    while (true){
        //get the index of the next element
        size_t min_index = inner.size;
        dtype total = 0;
        for (auto iter_index = 0; iter_index < component_count; iter_index++){
            if (comp_iters[iter_index] == comp_ends[iter_index])
                continue;
            auto v = *comp_iters[iter_index];
            if (v.first < min_index){
                min_index = v.first;
                total = v.second * comp_col_coffs[iter_index][min_index] * comp_row_coffs[iter_index] * factors[iter_index];
                min_set.clear();
                min_set.insert(iter_index);
            }
            else if (v.first == min_index){
                total += v.second * comp_col_coffs[iter_index][min_index] * comp_row_coffs[iter_index] * factors[iter_index];
                min_set.insert(iter_index);
            }
        }

        if (min_set.empty()){
            //we are done with this row
            break;
        }
        row_indices.push_back(min_index);
        row_data.push_back(total);
        inner.columns[min_index].insert(row_num);
        for (auto iter_to_advance: min_set){
            comp_iters[iter_to_advance]++;
        }
        min_set.clear();
    }
}

int binary_search(std::vector<size_t>& haystack, size_t needle){
    if (haystack.empty())
        return -1;
    int start = 0;
    int end = ((int)haystack.size())-1;
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

void ParasymbolicMatrix::rebuild_column(size_t col_num){
    auto col_set = inner.columns[col_num];
    for(auto row_num: col_set){
        dtype total = 0;
        for (auto comp_num = 0; comp_num < component_count; comp_num++){
            total += components[comp_num]->get(row_num, col_num) * factors[comp_num];
        }
        auto& row_indices = inner.indices[row_num];
        auto bin = binary_search(row_indices, col_num);
        inner.data[row_num][bin] = total;
    }
}
void ParasymbolicMatrix::rebuild_factor(dtype factor){
    for (auto row_num = 0; row_num < inner.size; row_num++){
        auto& row_data = inner.data[row_num];
        for (auto iter = row_data.begin(); iter != row_data.end(); ++iter){
            *iter *= factor;
        }
    }
}

dtype ParasymbolicMatrix::get(size_t row, size_t column){
    if (calc_lock)
        return NAN;
    auto& row_indices = inner.indices[row];
    auto bin = binary_search(row_indices, column);
    if (bin == -1)
        return 0;
    return inner.data[row][bin];
}
dtype ParasymbolicMatrix::get(size_t comp, size_t row, size_t column){
    if (calc_lock)
        return NAN;
    return components[comp]->get(row, column);
}
double ParasymbolicMatrix::total(){
    if (calc_lock)
        return NAN;
    dtype ret = 0;
    // todo kahan?
    for (auto i = 0; i < inner.size; i++){
        auto& data = inner.data[i];
        for (auto j = data.cbegin(); j < data.cend(); j++)
            ret += *j;
    }
    return ret;
}

void ParasymbolicMatrix::_prob_any_row(size_t row_num, dtype const* A_v, size_t v_len, size_t const * A_non_zero_indices, size_t nzi_len,
                        dtype** AF_out, size_t* o_size){
    dtype inv_ret = 1;
    size_t nz_index = 0;
    auto& row_indices = inner.indices[row_num];
    auto& row_data = inner.data[row_num];
    size_t r_i_index = 0;
    size_t t_i_len = row_indices.size();
    while (r_i_index != t_i_len && nz_index != nzi_len){
        auto i = row_indices[r_i_index];
        auto j = A_non_zero_indices[nz_index];
        if (i < j){
            r_i_index++;
        }
        else if (j < i){
            nz_index++;
        }
        else /*j == i*/{
            inv_ret *= (1 - row_data[r_i_index] * A_v[j]);
            r_i_index++;
            nz_index++;
        }
    }
    (*AF_out)[row_num] = 1-inv_ret;
}

void ParasymbolicMatrix::_prob_any(dtype const* A_v, size_t v_len, size_t const * A_non_zero_indices, size_t nzi_len,
                        dtype** AF_out, size_t* o_size){
    *o_size = inner.size;
    *AF_out = new dtype[inner.size];
    std::vector<std::future<void>> futures (inner.size);
    for (auto row_num = 0; row_num < inner.size; row_num++){
        futures[row_num] = pool->push([=](int) {
                this->_prob_any_row(row_num, A_v, v_len, A_non_zero_indices, nzi_len, AF_out, o_size);
            }
        );
    }
    for (auto row_num = 0; row_num < inner.size; row_num++){
        futures[row_num].get();
    }

}

void ParasymbolicMatrix::operator*=(dtype rhs){
    for (auto comp_num = 0; comp_num < component_count; comp_num++){
        factors[comp_num] *= rhs;
    }
    if (!calc_lock) rebuild_factor(rhs);
}

void ParasymbolicMatrix::set_factors(dtype const* A_factors, size_t f_len){
    for (auto comp_num = 0; comp_num < f_len; comp_num++){
        factors[comp_num] = A_factors[comp_num];
    }
    if (!calc_lock) rebuild_all();
}

void ParasymbolicMatrix::mul_sub_row(size_t component, size_t row, dtype factor){
    auto comp = components[component];
    comp->mul_row(row, factor);
    if (!calc_lock) rebuild_row(row);
}

void ParasymbolicMatrix::mul_sub_col(size_t component, size_t col, dtype factor){
    auto comp = components[component];
    comp->mul_col(col, factor);
    if (!calc_lock) rebuild_column(col);
}

void ParasymbolicMatrix::reset_mul_row(size_t component, size_t row){
    auto comp = components[component];
    comp->reset_mul_row(row);
    if (!calc_lock) rebuild_row(row);
}

void ParasymbolicMatrix::reset_mul_col(size_t component, size_t col){
    auto comp = components[component];
    comp->reset_mul_col(col);
    if (!calc_lock) rebuild_column(col);
}

void ParasymbolicMatrix::batch_set(size_t component_num, size_t row, size_t const* A_columns, size_t c_len,
         dtype const* A_values, size_t v_len){
    components[component_num]->batch_set(row, A_columns, c_len, A_values, v_len);
    if (!calc_lock) rebuild_row(row);
}

void ParasymbolicMatrix::set_calc_lock(bool value){
    calc_lock = value;
    // todo we probably don't have to rebuild the entire matrix
    if (!calc_lock) rebuild_all();
}

ParasymbolicMatrix::~ParasymbolicMatrix(){
    delete[] components;
    delete[] factors;
    delete pool;
}
// endregion