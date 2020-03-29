#include <map>
#include <vector>
#include <tuple>
#include <unordered_set>

using dtype = float;
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
        dtype operator[](size_t row,size_t column);
        void batch_set(size_t row, size_t const* columns, size_t c_len, dtype const* values, size_t v_len);
        dtype get_total();
        virtual ~BareSparseMatrix();
}

class CoffedSparseMatrix: public BareSparseMatrix{
    private:
        dtype* col_coefficients*;
        dtype* row_coefficients*;
        friend class ParasymbolicMatrix;
    public:
        CoffedSparseMatrix(int size);
        dtype operator[](size_t row, size_t column);
        void mul_row(size_t row, dtype factor);
        void mul_col(size_t col, dtype factor);
        void reset_mul_row(size_t row);
        void reset_mul_col(size_t col);
        virtual ~CoffedSparseMatrix();
}

class ParasymbolicMatrix{
    private:
        size_t component_count;
        dtype* factors;
        CoffedSparseMatrix* components;
        BareSparseMatrix inner;
        ParasymbolicMatrix(dtype const* factors, CoffedSparseMatrix const* comps, size_t len);

        void rebuild_all();
        void rebuild_row(size_t);
        void rebuild_column(size_t);
        void rebuild_factor(dtype);
    public:
        static ParasymbolicMatrix from_tuples(std::vector<std::tuple<CoffedSparseMatrix, dtype>>);
        dtype operator[](size_t row, size_t column);
        dtype total();
        void prob_any(dtype const* A_v, size_t v_len, size_t const * A_non_zero_indices, size_t nzi_len,
                        dtype** AF_out, size_t* o_size);
        ParasymbolicMatrix& operator*=(dtype rhs);
        void mul_sub_row(size_t component, size_t row, dtype factor);
        void mul_sub_col(size_t component, size_t col, dtype factor);
        virtual ~ParasymbolicMatrix();
}