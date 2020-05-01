#include "ctpl_stl.h"

#include <map>
#include <vector>
#include <tuple>
#include <unordered_set>

typedef float dtype;

using row_type = std::map<size_t, dtype>;
using col_type = std::unordered_set<size_t>;

class BareSparseMatrix{
    protected:
        // note, each column set only stores indices that might not be zero, the only guarantee
        // is that if in index is non-zero, it will appear in in the column set
        row_type* rows;
        col_type* columns;
        dtype total;
        friend class ParasymbolicMatrix;
        void set(size_t r, size_t c, dtype v);
    public:
        const size_t size;
        BareSparseMatrix(size_t size);
        dtype get(size_t row,size_t column);
        void batch_set(size_t row, size_t const* A_columns, size_t c_len, dtype const* A_values, size_t v_len);
        double get_total();
        virtual ~BareSparseMatrix();
};

class CoffedSparseMatrix: public BareSparseMatrix{
    private:
        dtype* col_coefficients;
        dtype* row_coefficients;
        friend class ParasymbolicMatrix;
    public:
        CoffedSparseMatrix(size_t size);
        dtype get(size_t row, size_t column);
        void mul_row(size_t row, dtype factor);
        void mul_col(size_t col, dtype factor);
        void set_row(size_t row, dtype coeff);
        void set_col(size_t row, dtype coeff);
        void reset_mul_row(size_t row);
        void reset_mul_col(size_t col);
        virtual ~CoffedSparseMatrix();

        std::vector<std::vector<size_t>> non_zero_columns();
        std::vector<size_t> non_zero_column(size_t row_num);
};

class FastSparseMatrix{
    private:
        std::vector<size_t>* indices;
        std::vector<dtype>* data;
        size_t size;
        col_type* columns;

        friend class ParasymbolicMatrix;
    public:
        FastSparseMatrix(size_t size);
        ~FastSparseMatrix();
};

class ParasymbolicMatrix{
    private:
        size_t component_count;
        dtype* factors;
        CoffedSparseMatrix** components;
        FastSparseMatrix inner;
        bool calc_lock;

        ctpl::thread_pool* pool;

        void rebuild_all();
        void rebuild_row(size_t);
        void rebuild_column(size_t);
        void rebuild_factor(dtype);

        void _prob_any_row(size_t row_num, dtype const* A_v, size_t v_len, size_t const * A_non_zero_indices, size_t nzi_len,
                        dtype** AF_out, size_t* o_size);
    public:
        ParasymbolicMatrix(size_t size, size_t component_count);
        dtype get(size_t row, size_t column);
        dtype get(size_t comp, size_t row, size_t column);
        double total();
        size_t get_size();
        void _prob_any(dtype const* A_v, size_t v_len, size_t const * A_non_zero_indices, size_t nzi_len,
                        dtype** AF_out, size_t* o_size);
        void operator*=(dtype rhs);
        void set_factors(dtype const* A_factors, size_t f_len);
        void mul_sub_row(size_t component, size_t row, dtype factor);
        void mul_sub_col(size_t component, size_t col, dtype factor);
        void reset_mul_row(size_t component, size_t row);
        void reset_mul_col(size_t component, size_t col);
        void set_sub_row(size_t component, size_t row, dtype coeff);
        void set_sub_col(size_t component, size_t col, dtype coeff);
        void batch_set(size_t component_num, size_t row, size_t const* A_columns, size_t c_len,
         dtype const* A_values, size_t v_len);
        void set_calc_lock(bool value);
        virtual ~ParasymbolicMatrix();
        std::vector<std::vector<std::vector<size_t>>> non_zero_columns();
        std::vector<std::vector<size_t>> non_zero_column(size_t row_num);
};