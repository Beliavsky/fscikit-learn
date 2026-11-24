program structured_samples
  use sklearn, only: dp,make_sparse_coded_signal,make_sparse_spd_matrix,sparse_coder,csr_matrix
  implicit none
  type(sparse_coder) :: model
  type(csr_matrix) :: precision
  real(dp), allocatable :: x(:,:),dictionary(:,:),code(:,:),encoded(:,:),dense(:,:)
  integer :: i
  call make_sparse_coded_signal(40,12,30,3,x,dictionary,code,random_state=42)
  model%dictionary=dictionary;model%transform_n_nonzero_coefs=3;encoded=model%transform(x)
  print *, 'Signal shape:',shape(x)
  print *, 'Dictionary shape:',shape(dictionary)
  print *, 'Code support:',minval(count(code/=0,dim=2)),maxval(count(code/=0,dim=2))
  print *, 'Sparse coder reconstruction error:',maxval(abs(x-matmul(encoded,dictionary)))
  call make_sparse_spd_matrix(precision,n_dim=12,alpha=.8_dp,norm_diag=.true.,random_state=42)
  dense=precision%to_dense()
  print *, 'Precision shape:',shape(dense)
  print *, 'Unit precision diagonal:',maxval(abs([(dense(i,i)-1,i=1,12)]))<1.e-14_dp
end program
