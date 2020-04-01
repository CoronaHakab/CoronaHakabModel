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
        void reset_mul_row(size_t row);
        void reset_mul_col(size_t col);
        virtual ~CoffedSparseMatrix();
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
        /**
        construct a parasymbolic matrix of shape size x size and component_count component matrices,
         all initialized to be zero
        */
        ParasymbolicMatrix(size_t size, size_t component_count);
        /**
        get the value of the matrix at cell [row, column]
        */
        dtype get(size_t row, size_t column);
        /**
        get the value of the comp component matrix at cell [row, column]
        */
        dtype get(size_t comp, size_t row, size_t column);
        /**
        get the sum total of all cell values
        */
        double total();
        /**
        set AF_out to be an array of the POA of the matrix at each row given an array v and the non-zero indices of v
        */
        void _prob_any(dtype const* A_v, size_t v_len, size_t const * A_non_zero_indices, size_t nzi_len,
                        dtype** AF_out, size_t* o_size);
        /**
        multiply all the values of the matrix in-place by a constant factor
        */
        void operator*=(dtype rhs);
        /**
        set the factors of the parasymbolic components (initially 1)
        undefined behaviour if the size of the factors is different than the number of components.
        */
        void set_factors(dtype const* A_factors, size_t f_len);
        /**
        multiply a row of a component matrix by a constant
        */
        void mul_sub_row(size_t component, size_t row, dtype factor);
        /**
        multiply a column of a component matrix by a constant
        */
        void mul_sub_col(size_t component, size_t col, dtype factor);
        /**
        reset a multiplier of a component matrix row to 1
        */
        void reset_mul_row(size_t component, size_t row);
        /**
        reset a multiplier of a component matrix column to 1
        */
        void reset_mul_col(size_t component, size_t col);
        /**
        set a row of a component matrix with a vector of column indices and values
        setting a row that has already been set is undefined behaviour
        */
        void batch_set(size_t component_num, size_t row, size_t const* A_columns, size_t c_len,
         dtype const* A_values, size_t v_len);
        /**
        set the calc_lock of the matrix, while the calc_lock is true, the cached matrix is not refreshed, and no queries can take place
        if the calc_lock is reset, the matrix will be recalculated
        */
        void set_calc_lock(bool value);
        virtual ~ParasymbolicMatrix();
};