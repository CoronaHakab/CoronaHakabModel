#include <map>
#include <vector>
#include <tuple>
#include <unordered_set>

typedef float dtype;

// todo benchmark

class MagicOperator{
    public:
        //invariant: if w_val = 0 or v_val = 0, operate(w_val, v_val) = 1
        virtual dtype operate(dtype w_val, dtype v_val) const;
        virtual MagicOperator* mul(dtype factor) const;
        virtual ~MagicOperator() = default;
};

class FactoredMagicOperator: public MagicOperator{
    private:
        MagicOperator const * inner;
        dtype factor;
    public:
        FactoredMagicOperator(MagicOperator const* inner, dtype factor);
        dtype operate(dtype w_val, dtype v_val) const;
        MagicOperator* mul(dtype factor) const;
};

class ManifestMatrix;

class SparseMatrix{
    private:
        // note, each column set only stores indices that might not be zero, the only guarantee
        // is that if in index is non-zero, it will appear in in the column set
        size_t* row_lens;
        size_t** row_indices;
        dtype** row_probs;
        dtype** row_values;
        dtype* prob_row_coefficients;
        dtype* prob_column_coefficients;
        dtype* value_row_offsets;
        dtype* value_column_offsets;
        size_t _nz_count;

        int col_ind(size_t row, size_t col) const;

        friend class ManifestMatrix;
    public:
        const size_t size;

        SparseMatrix(size_t size);
        void batch_set(size_t row, size_t const* A_columns, size_t c_len, dtype const* A_probs, size_t p_len, dtype const* A_values, size_t v_len);
        bool has_value(size_t row,size_t column) const;
        std::tuple<dtype, dtype> get(size_t row,size_t column) const;
        size_t nz_count();
        ManifestMatrix* manifest(dtype const* A_rolls, size_t r_len);

        void row_set_prob_coff(size_t row, dtype coff);
        void col_set_prob_coff(size_t column, dtype coff);

        void row_set_value_offset(size_t row, dtype offset);
        void col_set_value_offset(size_t column, dtype offset);

        std::vector<std::vector<size_t>> non_zero_columns();

        virtual ~SparseMatrix();
};

class ManifestMatrix{
    private:
        SparseMatrix const* origin;
        bool** is_manifest;
        ManifestMatrix(SparseMatrix* const & origin);

        friend class SparseMatrix;
    public:
        dtype get(size_t row,size_t column);
        void I_POA(dtype const* A_values, size_t v_len, size_t const* A_nz_indices, size_t nzi_len,
            MagicOperator* const& op,
            dtype** AF_out, size_t* o_len);
        std::vector<std::vector<size_t>> nz_rows();
        virtual ~ManifestMatrix();
};
//todo namespace?
const int mask_count = sizeof(dtype);

dtype from_bytes(unsigned char const* A_bytes, size_t b_len);

class WMaskedTabularOp: public MagicOperator{
    private:
        const dtype v_res;
        const size_t v_len;
        size_t mask_len[mask_count];
        size_t increments[mask_count];
        dtype* table;
    public:
        //invariant: if w_val = 0 or v_val = 0, operate(w_val, v_val) = 1
        WMaskedTabularOp(dtype v_res, size_t v_len, size_t const* A_mask_len, size_t ml_len,
                                                    dtype const * A_table, size_t table_len);
        virtual dtype operate(dtype w_val, dtype v_val) const;
        virtual ~WMaskedTabularOp();
};