! Dense/CSR RandomTreesEmbedding, adapted from sklearn 1.6.1 ensemble/_forest.py.
! BSD-3-Clause attribution retained in THIRD_PARTY_NOTICES.md.
module sklearn_tree_embedding
  use sklearn_common, only: dp,require,check_x
  use sklearn_tree, only: extra_tree_regressor
  use sklearn_tree_features, only: sample_count_control
  use sklearn_sparse, only: csr_matrix
  use sklearn_preprocessing, only: one_hot_encoder
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_ieee_math, only: ieee_mean,ieee_total,ieee_quotient
  use, intrinsic :: iso_fortran_env, only: real32,int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan
  implicit none
  private
  public :: random_trees_embedding
  type :: random_trees_embedding
    integer :: n_estimators=100,max_depth=5,min_samples_split=2,min_samples_leaf=1,random_state=0
    integer :: max_leaf_nodes=0
    real(dp) :: min_impurity_decrease=0,min_weight_fraction_leaf=0
    logical :: warm_start=.false.
    integer :: n_features_in_=0,n_features_out_=0,n_estimators_=0,n_samples_fit_=0,n_added_=0,n_outputs_=1
    type(extra_tree_regressor), allocatable :: estimators_(:)
    type(one_hot_encoder) :: one_hot_encoder_
    integer, allocatable :: estimator_seeds_(:),node_offsets_(:)
    real(dp), allocatable :: random_target_(:),feature_importances_(:)
    type(sample_count_control), private :: leaf_control_,split_control_
  contains
    procedure :: fit=>embedding_fit
    procedure :: transform=>embedding_transform
    procedure :: fit_transform=>embedding_fit_transform
    procedure :: transform_csr=>embedding_transform_csr
    procedure :: fit_transform_csr=>embedding_fit_transform_csr
    procedure :: apply=>embedding_apply
    procedure :: decision_path=>embedding_path
    procedure :: decision_path_csr=>embedding_path_csr
    procedure :: estimators_samples=>embedding_samples
    procedure :: get_feature_names_out=>embedding_names
    procedure, private :: leaf_integer,leaf_fraction,split_integer,split_fraction
    generic :: set_min_samples_leaf=>leaf_integer,leaf_fraction
    generic :: set_min_samples_split=>split_integer,split_fraction
  end type
contains
  subroutine leaf_integer(self,value)
    class(random_trees_embedding), intent(inout) :: self
    integer, intent(in) :: value
    self%min_samples_leaf=value; call self%leaf_control_%reset()
  end subroutine
  subroutine leaf_fraction(self,value)
    class(random_trees_embedding), intent(inout) :: self
    real(dp), intent(in) :: value
    self%min_samples_leaf=0; call self%leaf_control_%set(value)
  end subroutine
  subroutine split_integer(self,value)
    class(random_trees_embedding), intent(inout) :: self
    integer, intent(in) :: value
    self%min_samples_split=value; call self%split_control_%reset()
  end subroutine
  subroutine split_fraction(self,value)
    class(random_trees_embedding), intent(inout) :: self
    real(dp), intent(in) :: value
    self%min_samples_split=0; call self%split_control_%set(value)
  end subroutine
  subroutine embedding_fit(self,x,sample_weight)
    class(random_trees_embedding), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: sample_weight(:)
    type(extra_tree_regressor), allocatable :: trees(:)
    type(legacy_random_state) :: rng
    integer, allocatable :: leaves(:,:)
    real(dp), allocatable :: importances(:,:)
    real(dp) :: total
    integer :: n,p,old,i,j,ignored,count_split,depth,min_leaf,min_split
    call check_x(x); n=size(x,1); p=size(x,2)
    call require(self%n_estimators>0.and.self%n_estimators<huge(1),'invalid tree embedding estimator count')
    call require(self%random_state>=0,'tree embedding seed must be nonnegative')
    call require(self%max_depth>=-1,'invalid tree embedding depth')
    min_leaf=self%leaf_control_%resolve(self%min_samples_leaf,n,1,.false.)
    min_split=self%split_control_%resolve(self%min_samples_split,n,2,.true.)
    call require(self%max_leaf_nodes>=0.and.self%max_leaf_nodes/=1,'invalid tree embedding leaf budget')
    call require(ieee_is_finite(self%min_weight_fraction_leaf),'tree embedding weighted leaf fraction must be finite')
    call require(self%min_weight_fraction_leaf>=0.and.self%min_weight_fraction_leaf<=.5_dp, &
      'tree embedding weighted leaf fraction must be in [0,.5]')
    call require(ieee_is_finite(self%min_impurity_decrease),'tree embedding impurity decrease must be finite')
    call require(self%min_impurity_decrease>=0,'tree embedding impurity decrease must be nonnegative')
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'tree embedding weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'tree embedding weights must be finite')
    end if
    old=0
    if(self%warm_start.and.allocated(self%estimators_)) then
      old=size(self%estimators_)
      call require(p==self%n_features_in_,'warm tree embedding feature width changed')
      call require(self%n_estimators>=old,'warm tree embedding cannot remove estimators')
    end if
    call rng%seed(self%random_state)
    self%random_target_=[(rng%uniform(),i=1,n)]
    ! Forest tree seeds start from a fresh scalar-seeded RNG, not the state
    ! advanced while drawing random targets. Warm fits skip all old seed draws.
    call rng%seed(self%random_state)
    do i=1,old
      ignored=rng%integer_index(2147483647)
    end do
    allocate(trees(self%n_estimators))
    if(old>0) trees(:old)=self%estimators_
    depth=self%max_depth; if(depth==-1) depth=huge(1)
    do i=old+1,self%n_estimators
      trees(i)%random_state=rng%integer_index(2147483647)-1
      trees(i)%max_depth=depth; trees(i)%max_features=1
      trees(i)%min_samples_leaf=min_leaf; trees(i)%min_samples_split=min_split
      trees(i)%max_leaf_nodes=self%max_leaf_nodes; trees(i)%min_weight_fraction_leaf=self%min_weight_fraction_leaf
      trees(i)%min_impurity_decrease=self%min_impurity_decrease
      call trees(i)%fit(x,self%random_target_,sample_weight)
    end do
    call move_alloc(trees,self%estimators_)
    self%n_estimators_=size(self%estimators_); self%n_added_=self%n_estimators_-old
    self%n_features_in_=p; self%n_samples_fit_=n
    self%estimator_seeds_=[(self%estimators_(i)%random_state,i=1,self%n_estimators_)]
    if(allocated(self%node_offsets_)) deallocate(self%node_offsets_)
    allocate(self%node_offsets_(self%n_estimators_+1)); self%node_offsets_(1)=0
    do i=1,self%n_estimators_
      call require(self%estimators_(i)%node_count_<=huge(1)-self%node_offsets_(i), &
        'tree embedding node total exceeds integer range')
      self%node_offsets_(i+1)=self%node_offsets_(i)+self%estimators_(i)%node_count_
    end do
    leaves=self%apply(x)
    self%one_hot_encoder_=one_hot_encoder()
    call self%one_hot_encoder_%fit(real(leaves,dp)); self%n_features_out_=self%one_hot_encoder_%n_features_out_
    count_split=count(self%estimators_%node_count_>1)
    self%feature_importances_=spread(0._dp,1,p)
    if(count_split>0) then
      allocate(importances(count_split,p)); j=0
      do i=1,self%n_estimators_
        if(self%estimators_(i)%node_count_<=1) cycle
        j=j+1; importances(j,:)=self%estimators_(i)%feature_importances_
      end do
      do j=1,p
        self%feature_importances_(j)=ieee_mean(importances(:,j))
      end do
      total=ieee_total(self%feature_importances_)
      do j=1,p
        self%feature_importances_(j)=ieee_quotient(self%feature_importances_(j),total)
      end do
    end if
  end subroutine
  function embedding_apply(self,x) result(leaves)
    class(random_trees_embedding), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: leaves(:,:)
    integer :: i
    call require(allocated(self%estimators_),'tree embedding is not fitted')
    call require(size(x,1)>0.and.size(x,2)==self%n_features_in_,'tree embedding query shape mismatch')
    allocate(leaves(size(x,1),size(self%estimators_)))
    do i=1,size(self%estimators_)
      leaves(:,i)=self%estimators_(i)%apply(x)
    end do
  end function
  function embedding_transform(self,x) result(encoded)
    class(random_trees_embedding), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: encoded(:,:)
    integer, allocatable :: leaves(:,:)
    leaves=self%apply(x); encoded=self%one_hot_encoder_%transform(real(leaves,dp))
  end function
  function embedding_fit_transform(self,x,sample_weight) result(encoded)
    class(random_trees_embedding), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: encoded(:,:)
    call self%fit(x,sample_weight); encoded=self%transform(x)
  end function
  function embedding_transform_csr(self,x) result(encoded)
    class(random_trees_embedding), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    type(csr_matrix) :: encoded
    integer, allocatable :: leaves(:,:)
    leaves=self%apply(x); encoded=self%one_hot_encoder_%transform_csr(real(leaves,dp))
  end function
  function embedding_fit_transform_csr(self,x,sample_weight) result(encoded)
    class(random_trees_embedding), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    type(csr_matrix) :: encoded
    call self%fit(x,sample_weight); encoded=self%transform_csr(x)
  end function
  function embedding_path_csr(self,x,n_nodes_ptr) result(indicator)
    class(random_trees_embedding), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, optional, allocatable, intent(out) :: n_nodes_ptr(:)
    type(csr_matrix) :: indicator
    integer, allocatable :: checked(:)
    integer :: i,t,node,pass,position
    integer(int64) :: total
    call require(allocated(self%estimators_),'tree embedding is not fitted')
    call require(size(x,1)>0.and.size(x,2)==self%n_features_in_,'tree embedding query shape mismatch')
    checked=self%estimators_(1)%apply(x) ! Validate NaNs, infinities and float32 range using the tree backend.
    indicator%n_rows=size(x,1); indicator%n_cols=self%node_offsets_(size(self%node_offsets_))
    allocate(indicator%indptr(size(x,1)+1)); indicator%indptr(1)=1
    total=0
    do pass=1,2
      position=1
      do i=1,size(x,1)
        do t=1,size(self%estimators_)
          node=1
          do
            if(pass==1) then
              total=total+1
              call require(total<int(huge(1),int64),'tree embedding sparse path stored count overflow')
            else
              indicator%indices(position)=self%node_offsets_(t)+node; position=position+1
            end if
            if(self%estimators_(t)%nodes_(node)%left==0) exit
            node=path_child(self,t,node,x(i,:))
          end do
        end do
        if(pass==1) indicator%indptr(i+1)=int(total)+1
      end do
      if(pass==1) then
        allocate(indicator%indices(int(total)),indicator%data(int(total))); indicator%data=1
      end if
    end do
    if(present(n_nodes_ptr)) n_nodes_ptr=self%node_offsets_
  end function
  integer function path_child(self,t,node,row) result(child)
    class(random_trees_embedding), intent(in) :: self
    integer, intent(in) :: t,node
    real(dp), intent(in) :: row(:)
    integer :: f
    f=self%estimators_(t)%nodes_(node)%feature
    ! Source decision_path sends NaNs right independently of apply's learned route.
    child=self%estimators_(t)%nodes_(node)%right
    if(ieee_is_nan(row(f))) return
    if(real(real(row(f),real32),dp)<=self%estimators_(t)%nodes_(node)%threshold) &
      child=self%estimators_(t)%nodes_(node)%left
  end function
  function embedding_path(self,x,n_nodes_ptr) result(indicator)
    class(random_trees_embedding), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable, intent(out), optional :: n_nodes_ptr(:)
    integer, allocatable :: indicator(:,:),leaves(:,:)
    integer :: t,i,node
    leaves=self%apply(x) ! Reuse fitted-state, shape and float32-range checks.
    allocate(indicator(size(x,1),self%node_offsets_(size(self%node_offsets_)))); indicator=0
    do t=1,size(self%estimators_)
      do i=1,size(x,1)
        node=1
        do
          indicator(i,self%node_offsets_(t)+node)=1
          if(self%estimators_(t)%nodes_(node)%left==0) exit
          node=path_child(self,t,node,x(i,:))
        end do
      end do
    end do
    if(present(n_nodes_ptr)) n_nodes_ptr=self%node_offsets_
  end function
  function embedding_samples(self) result(samples)
    class(random_trees_embedding), intent(in) :: self
    integer, allocatable :: samples(:,:)
    integer :: i
    call require(allocated(self%estimators_),'tree embedding is not fitted')
    samples=spread([(i,i=1,self%n_samples_fit_)],2,size(self%estimators_))
  end function
  function embedding_names(self,input_features) result(names)
    class(random_trees_embedding), intent(in) :: self
    character(*), intent(in), optional :: input_features(:)
    character(64), allocatable :: names(:)
    integer :: i,j,k,leaf
    call require(allocated(self%estimators_),'tree embedding is not fitted')
    if(present(input_features)) call require(size(input_features)==self%n_features_in_, &
      'tree embedding input feature name count mismatch')
    allocate(names(self%n_features_out_)); k=0
    do i=1,size(self%estimators_)
      do j=1,size(self%one_hot_encoder_%categories_(i)%values)
        k=k+1; leaf=int(self%one_hot_encoder_%categories_(i)%values(j))-1
        write(names(k),'(a,i0,a,i0)') 'randomtreesembedding_',i-1,'_',leaf
      end do
    end do
  end function
end module
