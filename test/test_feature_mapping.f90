program test_feature_mapping
  use sklearn_feature_extraction
  use sklearn_murmurhash, only: murmurhash3_32
  use sklearn_common, only: dp,require
  use sklearn_sparse, only: csr_matrix
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  type(dict_vectorizer) :: vectorizer,copy
  type(feature_hasher) :: hasher
  type(feature_row) :: rows(3),tokens(2),empty(0)
  type(feature_row), allocatable :: inverse(:)
  type(feature_string), allocatable :: names(:)
  type(csr_matrix) :: matrix,other
  real(dp), allocatable :: dense(:,:)
  real(dp), parameter :: ill(4)=[1.e16_dp,1._dp,-1.e16_dp,.5_dp]
  integer :: i
  call require(murmurhash3_32('')==0,'empty MurmurHash3')
  call require(murmurhash3_32('foo')==-156908512_int64,'signed MurmurHash3 foo')
  call require(murmurhash3_32('hello')==613153351_int64,'MurmurHash3 hello')
  rows(1)%items=[feature_item(key='age',value=2._dp),feature_item(key='city',category='Paris')]
  rows(2)%items=[feature_item(key='age',value=3._dp),feature_item(key='city',category='Rome')]
  rows(3)%items=[feature_item(key='age ',value=4._dp),feature_item(key='city',category='Paris')]
  matrix=vectorizer%fit_transform(rows);dense=matrix%to_dense();names=vectorizer%get_feature_names_out()
  call require(all(shape(dense)==[3,4]),'dictionary feature shape')
  call require(vectorizer%vocabulary_index('age')==1.and.vectorizer%vocabulary_index('age ')==2, &
    'trailing spaces distinguish feature names')
  call require(all(dense(:,1)==[2._dp,3._dp,0._dp]),'numeric dictionary values')
  inverse=vectorizer%inverse_transform(matrix)
  other=vectorizer%transform(inverse)
  call require(all(other%to_dense()==dense),'flattened dictionary inverse round trip')
  copy=vectorizer;call copy%restrict([4,1]);other=copy%transform(rows)
  call require(other%n_cols==2.and.vectorizer%n_features_out_==4,'restriction order and copy independence')
  call copy%restrict([.false.,.false.]);other=copy%transform(rows)
  call require(other%n_cols==0.and.other%nnz()==0,'empty feature restriction')
  call copy%fit(empty);other=copy%transform(rows)
  call require(other%n_cols==0,'empty fit creates an empty vocabulary')
  tokens(1)%tokens=[feature_string('cat'),feature_string('cat'),feature_string('dog')]
  tokens(2)%tokens=[feature_string('dog')]
  hasher%n_features=1;hasher%input_type='string';hasher%alternate_sign=.false.
  matrix=hasher%fit_transform(tokens);dense=matrix%to_dense()
  call require(all(dense(:,1)==[3._dp,1._dp]),'hash collisions aggregate token counts')
  call require(matrix%nnz()==2,'hash CSR canonical entries')
  rows(1)%items=[feature_item(key='tag',categories=[feature_string('a'),feature_string('a')])]
  matrix=vectorizer%fit_transform(rows(:1))
  call require(matrix%nnz()==2,'dictionary repeated categories retain CSR duplicates')
  dense=matrix%to_dense();call require(dense(1,1)==2,'dictionary repeated categories count')
  rows(1)%items=[(feature_item(key='ill',value=ill(mod(i-1,4)+1)),i=1,48)]
  matrix=vectorizer%fit_transform(rows(:1));inverse=vectorizer%inverse_transform(matrix)
  call require(inverse(1)%items(1)%value==16._dp,'sparse inverse uses source NumPy pairwise sum')
  dense=matrix%to_dense();call require(dense(1,1)==.5_dp,'dense materialization uses CSR contribution order')
  print *, 'Dictionary vectorization and feature hashing: PASS'
end program
