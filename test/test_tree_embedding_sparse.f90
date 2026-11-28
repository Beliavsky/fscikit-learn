program test_tree_embedding_sparse
  use sklearn_tree_embedding, only: random_trees_embedding
  use sklearn_sparse, only: csr_matrix
  use sklearn_common, only: dp,require
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(random_trees_embedding) :: model,copy
  type(csr_matrix) :: encoded,path,again
  real(dp), allocatable :: x(:,:),w(:),dense(:,:)
  integer, allocatable :: pointer(:),indicator(:,:)
  integer :: i,t,j,n
  integer(int64) :: encoding_bytes,path_bytes
  character(32) :: mode
  call get_command_argument(1,mode)
  n=60; if(mode=='memory') n=2048
  allocate(x(n,3),w(n))
  do i=1,n
    x(i,:)=[sin(real(i,dp)),cos(.37_dp*i),real(i,dp)/n]
    w(i)=1+real(mod(i,5),dp)
  end do
  model%n_estimators=5; model%max_depth=4; model%random_state=7
  if(mode=='memory') then
    model%n_estimators=16; model%max_depth=-1
    encoded=model%fit_transform_csr(x)
    path=model%decision_path_csr(x,pointer)
    call encoded%validate(); call path%validate()
    call require(encoded%nnz()==n*model%n_estimators_,'large sparse embedding stored count')
    call require(all(encoded%data==1).and.all(path%data==1),'large sparse embedding binary storage')
    again=model%transform_csr(x)
    call require(all(again%indices==encoded%indices),'large sparse embedding repeated transform')
    encoding_bytes=int(n,int64)*encoded%n_cols*8; path_bytes=int(n,int64)*path%n_cols*4
    call require(encoding_bytes>256_int64*1024**2.and.path_bytes>256_int64*1024**2, &
      'large sparse embedding dense equivalents exceed memory limit')
    print *, 'dense_encoding_bytes',encoding_bytes
    print *, 'dense_path_bytes',path_bytes
    print *, 'Large sparse tree embedding memory case: PASS'
    stop
  end if
  select case(mode)
  case('leaf_budget'); model%max_leaf_nodes=1
  case('negative_budget'); model%max_leaf_nodes=-1
  case('weight_low'); model%min_weight_fraction_leaf=-.1_dp
  case('weight_high'); model%min_weight_fraction_leaf=.51_dp
  case('weight_nan'); model%min_weight_fraction_leaf=ieee_value(0._dp,ieee_quiet_nan)
  case('leaf_zero'); call model%set_min_samples_leaf(0._dp)
  case('leaf_one'); call model%set_min_samples_leaf(1._dp)
  case('split_zero'); call model%set_min_samples_split(0._dp)
  case('split_high'); call model%set_min_samples_split(1.1_dp)
  case('unfitted'); encoded=model%transform_csr(x)
  case('unfitted_path'); path=model%decision_path_csr(x)
  end select
  encoded=model%fit_transform_csr(x,w); dense=model%transform(x)
  call require(all(encoded%to_dense()==dense),'CSR embedding equals dense codes')
  call require(all(encoded%data==1).and.encoded%nnz()==n*5,'CSR embedding binary stored count')
  path=model%decision_path_csr(x,pointer); indicator=model%decision_path(x)
  call require(all(path%to_dense()==real(indicator,dp)),'CSR embedding path equals dense')
  call require(all(pointer==model%node_offsets_),'CSR embedding path offsets')
  if(mode=='features') then
    encoded=model%transform_csr(x(:,:2))
  else if(mode=='infinite_path') then
    x(1,1)=ieee_value(0._dp,ieee_positive_inf); path=model%decision_path_csr(x)
  end if
  x(1,:)=ieee_value(0._dp,ieee_quiet_nan)
  path=model%decision_path_csr(x); indicator=model%decision_path(x)
  call require(all(path%to_dense()==real(indicator,dp)),'CSR embedding path preserves Source NaN routing')
  x(1,:)=[sin(1._dp),cos(.37_dp),1._dp/n]
  model%max_leaf_nodes=3; model%max_depth=-1; model%min_weight_fraction_leaf=.15_dp
  call model%set_min_samples_leaf(.1_dp); call model%set_min_samples_split(.2_dp)
  encoded=model%fit_transform_csr(x,w)
  do t=1,model%n_estimators_
    call require(model%estimators_(t)%get_n_leaves()<=3,'embedding best-first leaf budget')
    do j=1,model%estimators_(t)%node_count_
      if(model%estimators_(t)%nodes_(j)%left/=0) cycle
      call require(model%estimators_(t)%nodes_(j)%n_samples>=6,'embedding fractional minimum leaf size')
      call require(model%estimators_(t)%nodes_(j)%weighted_n_samples>=.15_dp*sum(w), &
        'embedding minimum weighted leaf mass')
    end do
  end do
  copy=model
  model%warm_start=.true.; model%n_estimators=7; call model%set_min_samples_leaf(1)
  call model%set_min_samples_split(2); call model%fit(x)
  do t=1,5
    call require(model%estimators_(t)%node_count_==copy%estimators_(t)%node_count_,'warm CSR embedding prefix')
  end do
  model%one_hot_encoder_%handle_unknown='ignore'; model%one_hot_encoder_%drop='first'
  model%one_hot_encoder_%max_categories=2; call model%fit(x)
  encoded=model%transform_csr(x)
  call require(encoded%nnz()==n*7.and.model%one_hot_encoder_%drop=='none', &
    'embedding refit resets the complete one-hot encoder')
  print *, 'Sparse RandomTreesEmbedding: PASS'
end program
