program reference_tree_embedding_driver
  use sklearn_tree_embedding
  use sklearn_sparse, only: csr_matrix
  use sklearn_common, only: dp
  implicit none
  type(random_trees_embedding) :: model
  type(csr_matrix) :: csr_train,csr_query,csr_path
  character(32) :: mode
  real(dp) :: leaf_fraction,split_fraction
  integer :: leaf_kind,split_kind
  real(dp), allocatable :: x(:,:),query(:,:),weight(:),encoded(:,:),prediction(:,:),categories(:)
  integer, allocatable :: leaves(:,:),query_leaves(:,:),indicator(:,:),pointer(:),samples(:,:),stats(:,:),offsets(:)
  character(64), allocatable :: names(:)
  character(1024) :: input,output
  character(32) :: prefix,key
  integer :: u,v,stages,stage,n,p,m,warm,weighted,method,i,j,k
  call get_command_argument(1,input); call get_command_argument(2,output)
  call get_command_argument(3,mode)
  open(newunit=u,file=trim(input),status='old',action='read')
  open(newunit=v,file=trim(output),status='replace',action='write'); read(u,*) stages
  do stage=1,stages
    read(u,*) n,p,m,model%n_estimators,model%max_depth,model%min_samples_leaf,model%min_samples_split, &
      model%random_state,warm,weighted,method,model%min_impurity_decrease
    if(mode=='extended') then
      read(u,*) model%max_leaf_nodes,model%min_weight_fraction_leaf,leaf_kind,leaf_fraction,split_kind,split_fraction
      if(leaf_kind==1) call model%set_min_samples_leaf(leaf_fraction)
      if(split_kind==1) call model%set_min_samples_split(split_fraction)
    end if
    model%warm_start=warm==1
    if(allocated(x)) deallocate(x,query)
    if(allocated(weight)) deallocate(weight)
    allocate(x(n,p),query(m,p))
    do i=1,n
      read(u,*) x(i,:)
    end do
    if(weighted==1) then
      allocate(weight(n)); read(u,*) weight
    end if
    do i=1,m
      read(u,*) query(i,:)
    end do
    if(method==3) then
      model%one_hot_encoder_%drop='first'; model%one_hot_encoder_%handle_unknown='ignore'
      model%one_hot_encoder_%max_categories=2
    end if
    if(method==2) then
      csr_train=model%fit_transform_csr(x,weight); encoded=csr_train%to_dense()
    else if(method==0) then
      encoded=model%fit_transform(x,weight)
    else
      call model%fit(x,weight); encoded=model%transform(x)
    end if
    if(method/=2) csr_train=model%transform_csr(x)
    csr_query=model%transform_csr(query); csr_path=model%decision_path_csr(query)
    prediction=model%transform(query); leaves=model%apply(x); query_leaves=model%apply(query)
    indicator=model%decision_path(query,pointer); samples=model%estimators_samples(); names=model%get_feature_names_out()
    write(prefix,'(a,i0,a)') 's',stage,'_'
    call emit_csr('csr_train',csr_train); call emit_csr('csr_query',csr_query); call emit_csr('csr_path',csr_path)
    call emit('train',reshape(encoded,[size(encoded)])); call emit('query',reshape(prediction,[size(prediction)]))
    call emit('leaves',real(reshape(leaves,[size(leaves)]),dp))
    call emit('query_leaves',real(reshape(query_leaves,[size(query_leaves)]),dp))
    call emit('path',real(reshape(indicator,[size(indicator)]),dp)); call emit('pointer',real(pointer,dp))
    call emit('samples',real(reshape(samples,[size(samples)]),dp)); call emit('seeds',real(model%estimator_seeds_,dp))
    call emit('random_target',model%random_target_); call emit('importances',model%feature_importances_)
    call emit('counts',real([model%n_features_in_,model%n_features_out_,model%n_estimators_,model%n_samples_fit_, &
      model%n_added_,model%n_outputs_],dp))
    if(allocated(stats)) deallocate(stats,offsets)
    allocate(stats(model%n_estimators_,4),offsets(model%n_estimators_+1)); offsets(1)=0
    categories=[real(dp)::]
    do i=1,model%n_estimators_
      stats(i,:)=[model%estimators_(i)%node_count_,model%estimators_(i)%get_depth(), &
        model%estimators_(i)%get_n_leaves(),model%estimators_(i)%max_features_]
      categories=[categories,model%one_hot_encoder_%categories_(i)%values]
      offsets(i+1)=size(categories)
    end do
    call emit('tree_stats',real(reshape(stats,[size(stats)]),dp)); call emit('categories',categories)
    call emit('category_offsets',real(offsets,dp))
    key=trim(prefix)//'names'; write(v,'(a,*(1x,a))') trim(key),(trim(names(i)),i=1,size(names))
  end do
  close(u); close(v)
contains
  subroutine emit_csr(name,matrix)
    character(*), intent(in) :: name
    type(csr_matrix), intent(in) :: matrix
    call emit(name//'_shape',real([matrix%n_rows,matrix%n_cols],dp))
    call emit(name//'_indptr',real(matrix%indptr,dp))
    call emit(name//'_indices',real(matrix%indices,dp)); call emit(name//'_data',matrix%data)
  end subroutine
  subroutine emit(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:)
    write(v,'(a,*(1x,es27.17e3))') trim(prefix)//name,values
  end subroutine
end program
