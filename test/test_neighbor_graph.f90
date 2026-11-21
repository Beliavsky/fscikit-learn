program test_neighbor_graph
  use sklearn, only: dp,csr_matrix,csr_from_arrays,csr_from_dense, &
    kneighbors_transformer,radius_neighbors_transformer
  use sklearn_common, only: require
  implicit none
  type(kneighbors_transformer) :: kn,copy
  type(radius_neighbors_transformer) :: rn
  type(csr_matrix) :: g,s,t
  real(dp) :: x(4,1),d(4,4)
  real(dp), allocatable :: a(:,:),b(:,:)
  logical, allocatable :: mask(:,:)
  character(48), allocatable :: names(:)
  integer :: i
  x(:,1)=[0._dp,1._dp,3._dp,7._dp]
  kn%n_neighbors=1; g=kn%fit_transform(x)
  call require(g%nnz()==8.and.all(g%indptr==[1,3,5,7,9]),'distance graphs request k+1 edges')
  call require(count(g%data==0)==4,'distance graphs retain explicit diagonal zeros')
  mask=g%stored_mask(); call require(all([(mask(i,i),i=1,4)]),'CSR mask preserves zero-distance edges')
  names=kn%get_feature_names_out()
  call require(trim(names(4))=='kneighborstransformer3','graph output feature names')
  copy=kn; kn%mode='connectivity'; g=kn%transform(x)
  call require(g%nnz()==4.and.all(g%data==1),'connectivity requests k neighbors')
  g=copy%transform(); call require(all(g%data>0),'implicit training queries exclude self')
  rn%radius=2; g=rn%fit_transform(x)
  call require(g%nnz()==8.and.count(g%data==0)==4,'radius includes self and inclusive boundary')
  g=rn%transform(); call require(g%nnz()==4,'implicit radius queries remove self')
  rn%radius=0; rn%mode='connectivity'; g=rn%transform(x)
  call require(g%nnz()==4,'zero radius retains exact matches')
  d=0
  do i=1,4
    d(i,:)=abs(x(i,1)-x(:,1))
  end do
  s=csr_from_dense(d,.true.); kn%metric='precomputed'; kn%mode='distance'
  g=kn%fit_transform(s); t=copy%transform(x)
  call require(all(g%indices==t%indices).and.all(g%data==t%data),'CSR and feature distances agree')
  ! Missing entries and stored zeros are distinct; repeated columns are retained.
  s=csr_from_arrays(2,2,[1,4,6],[2,1,2,2,1],[2._dp,0._dp,1._dp,0._dp,3._dp])
  kn%n_neighbors=1; g=kn%fit_transform(s)
  call require(all(g%indices==[1,2,2,1]).and.all(g%data==[0._dp,1._dp,0._dp,3._dp]), &
    'CSR precomputed distances sort stored entries without coalescing')
  call require(all(s%data==[2._dp,0._dp,1._dp,0._dp,3._dp]),'fit preserves caller CSR')
  a=s%to_dense(); call require(all(a==reshape([0._dp,3._dp,3._dp,0._dp],[2,2])),'CSR dense sums duplicates')
  b=s%matmul(reshape([1._dp,2._dp],[2,1])); call require(all(b(:,1)==[6._dp,3._dp]),'CSR matmul')
  rn%metric='precomputed'; rn%mode='distance'; rn%radius=0; g=rn%fit_transform(s)
  call require(g%nnz()==2.and.all(g%data==0),'CSR radius zero preserves stored exact edges')
  rn%radius=.1_dp; s=csr_from_arrays(1,2,[1,1],[integer::],[real(dp)::]); g=rn%transform(s)
  call require(g%nnz()==0.and.all(g%indptr==1),'empty radius rows are valid')
  kn%metric='euclidean'; call kn%fit(x)
  g=kn%transform(x)
  call require(g%n_cols==4.and.kn%n_features_in_==1,'refit replaces CSR training state')
  call require(copy%n_samples_fit_==4,'independent fitted value copy')
  print *, 'Neighbor graph transformers and CSR: PASS'
end program
