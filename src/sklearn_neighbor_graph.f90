! Native graph transformers following sklearn.neighbors._graph / _base.
! See THIRD_PARTY_NOTICES.md for the scikit-learn BSD attribution.
module sklearn_neighbor_graph
  use sklearn_common, only: dp, require, check_x, argsort
  use sklearn_metrics, only: pairwise_distances
  use sklearn_sparse
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_is_nan
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  private
  public :: kneighbors_transformer, radius_neighbors_transformer
  type, abstract :: graph_transformer
    character(16) :: mode='distance', metric='minkowski'
    real(dp) :: p=2
    integer :: n_features_in_=0,n_features_out_=0,n_samples_fit_=0
    real(dp), allocatable, private :: x_(:,:)
    type(csr_matrix), private :: graph_
    logical, private :: sparse_precomputed_=.false.
    character(16), private :: metric_='minkowski'
    real(dp), private :: p_=2
  contains
    procedure :: fit_dense
    procedure :: fit_sparse
    generic :: fit=>fit_dense,fit_sparse
    procedure :: transform_dense
    procedure :: transform_sparse
    generic :: transform=>transform_dense,transform_sparse
    procedure :: fit_transform_dense
    procedure :: fit_transform_sparse
    generic :: fit_transform=>fit_transform_dense,fit_transform_sparse
    procedure :: get_feature_names_out=>feature_names
  end type
  type, extends(graph_transformer) :: kneighbors_transformer
    integer :: n_neighbors=5
  end type
  type, extends(graph_transformer) :: radius_neighbors_transformer
    real(dp) :: radius=1
  end type
contains
  subroutine parameters(self)
    class(graph_transformer), intent(in) :: self
    call require(self%mode=='distance'.or.self%mode=='connectivity','invalid neighbor graph mode')
    call require(any(self%metric==[character(16)::'minkowski','euclidean','sqeuclidean','manhattan', &
      'cityblock','chebyshev','cosine','precomputed']),'unsupported neighbor graph metric')
    call require(ieee_is_finite(self%p),'neighbor graph p must be finite')
    call require(self%p>=1,'neighbor graph p must be at least one')
    select type(self)
    type is(kneighbors_transformer)
      call require(self%n_neighbors>0.and.self%n_neighbors<huge(1)-1,'invalid graph n_neighbors')
    type is(radius_neighbors_transformer)
      call require(.not.ieee_is_nan(self%radius),'radius is NaN')
      call require(self%radius>=0,'radius must be nonnegative')
    end select
  end subroutine
  subroutine fit_dense(self,x)
    class(graph_transformer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    call parameters(self); call check_x(x)
    if(self%metric=='precomputed') then
      call require(size(x,1)==size(x,2),'precomputed fit matrix must be square')
      call require(all(x>=0),'precomputed distances must be nonnegative')
    end if
    self%x_=x; self%sparse_precomputed_=.false.
    self%graph_=csr_matrix()
    self%metric_=self%metric; self%p_=self%p
    self%n_features_in_=size(x,2); self%n_samples_fit_=size(x,1); self%n_features_out_=size(x,1)
  end subroutine
  subroutine fit_sparse(self,x)
    class(graph_transformer), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    call parameters(self); call x%validate()
    call require(self%metric=='precomputed','CSR neighbor input currently requires metric=precomputed')
    call require(x%n_rows>0.and.x%n_cols==x%n_rows,'precomputed CSR fit matrix must be nonempty and square')
    call require(all(ieee_is_finite(x%data)),'precomputed CSR contains nonfinite distances')
    call require(all(x%data>=0),'precomputed CSR distances must be nonnegative')
    self%graph_=x; self%sparse_precomputed_=.true.
    if(allocated(self%x_)) deallocate(self%x_)
    self%metric_=self%metric; self%p_=self%p
    self%n_features_in_=x%n_cols; self%n_samples_fit_=x%n_rows; self%n_features_out_=x%n_rows
  end subroutine
  function transform_dense(self,x,self_query) result(graph)
    class(graph_transformer), intent(in) :: self
    real(dp), intent(in), optional :: x(:,:)
    logical, intent(in), optional :: self_query
    type(csr_matrix) :: graph,query
    real(dp), allocatable :: distances(:,:)
    logical :: same
    integer :: i
    call require(self%n_samples_fit_>0,'neighbor transformer is not fitted'); call parameters(self)
    if(.not.present(x)) then
      if(self%sparse_precomputed_) then
        query=self%graph_
      else if(self%metric_=='precomputed') then
        query=csr_from_dense(self%x_,.true.)
      else
        distances=pairwise_distances(self%x_,self%x_,self%metric_,self%p_)
        do i=1,size(distances,1)
          distances(i,i)=0
        end do
        query=csr_from_dense(distances,.true.)
      end if
    else
      call check_x(x); call require(size(x,2)==self%n_features_in_,'neighbor query feature count mismatch')
      if(self%metric_=='precomputed') then
        call require(all(x>=0),'precomputed query distances must be nonnegative')
        query=csr_from_dense(x,.true.)
      else
        distances=pairwise_distances(x,self%x_,self%metric_,self%p_)
        same=.false.; if(present(self_query)) same=self_query
        if(same) then
          call require(all(shape(x)==shape(self%x_)),'self_query requires training shape')
          call require(all(x==self%x_),'self_query requires training values')
          do i=1,size(distances,1)
            distances(i,i)=0
          end do
        end if
        query=csr_from_dense(distances,.true.)
      end if
    end if
    graph=select_graph(self,query,.not.present(x),.false.)
  end function
  function transform_sparse(self,x) result(graph)
    class(graph_transformer), intent(in) :: self
    type(csr_matrix), intent(in) :: x
    type(csr_matrix) :: graph
    call require(self%n_samples_fit_>0,'neighbor transformer is not fitted'); call parameters(self)
    call require(self%metric_=='precomputed','CSR neighbor query requires metric=precomputed')
    graph=select_graph(self,x,.false.,.true.)
  end function
  function select_graph(self,query,exclude_self,sparse_order) result(graph)
    class(graph_transformer), intent(in) :: self
    type(csr_matrix), intent(in) :: query
    logical, intent(in) :: exclude_self,sparse_order
    type(csr_matrix) :: graph
    integer, allocatable :: order(:),cols(:)
    real(dp), allocatable :: vals(:)
    integer :: i,j,k,n,take,used,removed,capacity,slot
    integer(int64) :: capacity64
    call query%validate()
    call require(query%n_rows>0.and.query%n_cols==self%n_samples_fit_,'invalid neighbor graph query shape')
    call require(all(ieee_is_finite(query%data)),'neighbor distances exceed finite working range')
    call require(all(query%data>=0),'neighbor distances must be nonnegative')
    capacity64=size(query%data,kind=int64)
    select type(self)
    type is(kneighbors_transformer)
      take=self%n_neighbors; if(self%mode=='distance') take=take+1
      capacity64=int(query%n_rows,int64)*take
    end select
    call require(capacity64<huge(1),'neighbor graph exceeds integer capacity'); capacity=int(capacity64)
    graph%n_rows=query%n_rows; graph%n_cols=query%n_cols
    allocate(graph%indptr(query%n_rows+1),cols(capacity),vals(capacity)); used=0; graph%indptr(1)=1
    do i=1,query%n_rows
      n=query%indptr(i+1)-query%indptr(i)
      order=argsort(query%data(query%indptr(i):query%indptr(i+1)-1))
      select type(self)
      type is(kneighbors_transformer)
        take=self%n_neighbors; if(self%mode=='distance') take=take+1
        k=take; if(exclude_self) k=k+1
        call require(k<=self%n_samples_fit_,'not enough stored neighbors or fitted samples for requested graph')
        call require(n>=k,'not enough stored neighbors for requested graph')
        ! sklearn removes self, or the first selected entry if self was not selected.
        removed=0
        if(exclude_self) then
          do j=1,k
            slot=query%indptr(i)+order(j)-1
            if(query%indices(slot)==i) then
              call require(removed==0,'duplicate self entries are unsupported in implicit training query')
              removed=j
            end if
          end do
          if(removed==0) removed=1
        end if
        do j=1,k
          if(j==removed) cycle
          slot=query%indptr(i)+order(j)-1
          used=used+1; cols(used)=query%indices(slot); vals(used)=query%data(slot)
        end do
      type is(radius_neighbors_transformer)
        do j=1,n
          slot=query%indptr(i)+j-1
          if(self%mode=='distance'.or.sparse_order.or.self%sparse_precomputed_) &
            slot=query%indptr(i)+order(j)-1
          if(exclude_self.and.query%indices(slot)==i) cycle
          if(query%data(slot)>self%radius) cycle
          used=used+1; cols(used)=query%indices(slot); vals(used)=query%data(slot)
        end do
      end select
      graph%indptr(i+1)=used+1
    end do
    graph%indices=cols(:used); graph%data=vals(:used)
    if(self%mode=='connectivity') graph%data=1
  end function
  function fit_transform_dense(self,x) result(graph)
    class(graph_transformer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(csr_matrix) :: graph
    call self%fit(x); graph=self%transform(x,self_query=.true.)
  end function
  function fit_transform_sparse(self,x) result(graph)
    class(graph_transformer), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    type(csr_matrix) :: graph
    call self%fit(x); graph=self%transform(x)
  end function
  function feature_names(self) result(names)
    class(graph_transformer), intent(in) :: self
    character(48), allocatable :: names(:)
    character(32) :: prefix
    integer :: i
    call require(self%n_samples_fit_>0,'neighbor transformer is not fitted')
    select type(self)
    type is(kneighbors_transformer)
      prefix='kneighborstransformer'
    type is(radius_neighbors_transformer)
      prefix='radiusneighborstransformer'
    end select
    allocate(names(self%n_features_out_))
    do i=1,size(names)
      write(names(i),'(a,i0)') trim(prefix),i-1
    end do
  end function
end module
