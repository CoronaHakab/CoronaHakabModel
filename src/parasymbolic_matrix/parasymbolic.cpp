#include "parasymbolic.hpp"
#include <math.h>

// region bare
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
    auto row = rows[row_num];
    auto iter = row.begin();
    auto tup_index = 0;
    while (true){
        auto next_i_col = columns[tup_index];
        auto next_i_val = values[tup_index];
        if ((next_i_val == 0) && ++tup_index == c_len) return;

        while (iter->first < next_i_col){
            if (++iter == row.end()) break;
        }
        if (iter == row.end()){
            for (auto i = tup_index; i < c_len; ++i){
                row.insert({columns[i], values[i]});
                this->columns[columns[i]].insert(row_num);
                this->total += next_i_val;
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

dtype BareSparseMatrix::get_total(){
    if (!isnan(total)){
        return total;
    }
    // todo kahan sum?
    total = 0;
    for (auto row_ind = 0; row_ind < size; row_ind++){
        auto row = rows[row_ind];
        for (auto pair: row){
            total += pair.second;
        }
    }
    return total;
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
ParasymbolicMatrix::ParasymbolicMatrix(dtype* factors, CoffedSparseMatrix* const* comps, size_t len):
 component_count(len), factors(factors), components(components), inner(comps[0]->size){
    rebuild_all();
}

void ParasymbolicMatrix::rebuild_all(){
    inner.total = NAN;
    // todo parallelize
    for(auto row_num = 0; row_num < inner.size; row_num++){
        rebuild_row(row_num);
    }
}

void ParasymbolicMatrix::rebuild_row(size_t row_num){
    // todo most of the time, we'll only need to reset some cells in each row (only non-zero in the changed cells)
    row_type& row = inner.rows[row_num];
    decltype(row.begin())* comp_iters = new decltype(row.begin())[component_count];
    decltype(row.end())* comp_ends = new decltype(row.end())[component_count];
    dtype** comp_col_coffs = new dtype*[component_count];
    dtype* comp_row_coffs = new dtype[component_count];
    for (auto comp_num = 0; comp_num < component_count; comp_num++){
        comp_ends[comp_num] = components[comp_num]->rows[row_num].end();
        if ((comp_row_coffs[comp_num] = components[comp_num]->row_coefficients[row_num]) == 0){
            comp_iters[comp_num] = comp_ends[comp_num];
        }
        else{
            comp_iters[comp_num] = components[comp_num]->rows[row_num].begin();
            comp_col_coffs[comp_num] = components[comp_num]->col_coefficients;
        }
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
                total = v.second * comp_col_coffs[iter_index][min_index] * comp_row_coffs[iter_index];
                min_set.clear();
                min_set.insert(iter_index);
            }
            else if (v.first == min_index){
                total += v.second * comp_col_coffs[iter_index][min_index] * comp_row_coffs[iter_index];
                min_set.insert(iter_index);
            }
        }

        if (min_set.empty()){
            //we are done with this row
            break;
        }
        row.insert(row.end(), {min_index, total});
        inner.columns[min_index].insert(row_num);
        for (auto iter_to_advance: min_set){
            comp_iters[iter_to_advance]++;
        }
        min_set.clear();
    }
}
void ParasymbolicMatrix::rebuild_column(size_t col_num){
    auto col_set = inner.columns[col_num];
    for(auto row_num: col_set){
        dtype total = 0;
        for (auto comp_num = 0; comp_num < component_count; comp_num++){
            total += components[comp_num]->get(row_num, col_num);
        }
        inner.set(row_num, col_num, total);
    }
}

void ParasymbolicMatrix::rebuild_factor(dtype factor){
    for (auto row_num = 0; row_num < inner.size; row_num++){
        auto row = inner.rows[row_num];
        for (auto iter = row.begin(); iter != row.end(); ++iter){
            iter->second *= factor;
        }
    }
    inner.total *= factor;
}

ParasymbolicMatrix* ParasymbolicMatrix::from_tuples(std::vector<std::tuple<CoffedSparseMatrix*, dtype>> members){
    auto comps = new CoffedSparseMatrix* [members.size()];
    auto factors = new dtype[members.size()];

    for (auto tup_index = 0; tup_index < members.size(); tup_index++){
        comps[tup_index] = std::get<0>(members[tup_index]);
        factors[tup_index] = std::get<1>(members[tup_index]);
    }

    return new ParasymbolicMatrix(factors, comps, members.size());
}

dtype ParasymbolicMatrix::get(size_t row, size_t column){
    return inner.get(row, column);
}
dtype ParasymbolicMatrix::total(){
    return inner.get_total();
}

void ParasymbolicMatrix::prob_any(dtype const* A_v, size_t v_len, size_t const * A_non_zero_indices, size_t nzi_len,
                        dtype** AF_out, size_t* o_size){
    *o_size = inner.size;
    *AF_out = new dtype[inner.size];
    // todo parallelize
    for (auto row_num = 0; row_num < inner.size; row_num++){
        dtype inv_ret = 1;
        size_t nz_index = 0;
        auto row = inner.rows[row_num];
        auto r_iter = row.cbegin();
        while (r_iter != row.cend() && nz_index != nzi_len){
            auto i = r_iter->first;
            auto j = A_non_zero_indices[nz_index];
            if (i < j){
                r_iter++;
            }
            else if (j < i){
                nz_index++;
            }
            else /*j == i*/{
                inv_ret *= (1 - r_iter->second * A_v[j]);
                r_iter++;
                nz_index++;
            }
        }
        *AF_out[row_num] = 1-inv_ret;
    }
}

ParasymbolicMatrix& ParasymbolicMatrix::operator*=(dtype rhs){
    for (auto comp_num = 0; comp_num < component_count; comp_num++){
        factors[comp_num] *= rhs;
    }
    rebuild_factor(rhs);
    return *this;
}

void ParasymbolicMatrix::mul_sub_row(size_t component, size_t row, dtype factor){
    auto comp = components[component];
    comp->mul_row(row, factor);
    rebuild_row(row);
}

void ParasymbolicMatrix::mul_sub_col(size_t component, size_t col, dtype factor){
    auto comp = components[component];
    comp->mul_col(col, factor);
    rebuild_column(col);
}

void ParasymbolicMatrix::reset_mul_row(size_t component, size_t row){
    auto comp = components[component];
    comp->reset_mul_row(row);
    rebuild_column(row);
}

void ParasymbolicMatrix::reset_mul_col(size_t component, size_t col){
    auto comp = components[component];
    comp->reset_mul_col(col);
    rebuild_column(col);
}

ParasymbolicMatrix::~ParasymbolicMatrix(){
    delete components;
    delete factors;
}
// endregion