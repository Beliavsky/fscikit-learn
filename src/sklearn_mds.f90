! Dense SMACOF following sklearn/manifold/_mds.py, scikit-learn 1.6.1.
! See THIRD_PARTY_NOTICES.md for upstream BSD-3-Clause attribution.
module sklearn_mds
  use sklearn_common
  use sklearn_smacof,only:smacof,smacof_result
  use sklearn_pairwise_chunks,only:pairwise_distances_chunked,pairwise_distance_iterator,distance_chunk
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: mds
  type :: mds
    integer :: n_components=2,n_init=4,max_iter=300,random_state=0,n_iter_=0
    logical :: metric=.true.,converged_=.false.
    real(dp) :: eps=1.e-3_dp,stress_=0
    character(16) :: dissimilarity='euclidean'
    character(8) :: normalized_stress='auto'
    real(dp), allocatable :: embedding_(:,:),dissimilarity_matrix_(:,:)
  contains
    procedure :: fit=>mds_fit
    procedure :: fit_transform=>mds_fit_transform
  end type
contains
  subroutine mds_fit(self,x,init)
    class(mds), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: init(:,:)
    type(smacof_result)::fit
    type(pairwise_distance_iterator)::stream
    type(distance_chunk)::chunk
    integer::n,k,i
    logical::normalized,ok
    call check_x(x); n=size(x,1); k=self%n_components
    call require(n>=2,'MDS requires at least two samples')
    call require(k>0.and.self%n_init>0.and.self%max_iter>0,'invalid MDS iteration/dimension controls')
    call require(ieee_is_finite(self%eps),'invalid MDS tolerance')
    call require(self%eps>=0,'invalid MDS tolerance')
    call require(self%normalized_stress=='auto'.or.self%normalized_stress=='true'.or. &
      self%normalized_stress=='false','invalid MDS normalized_stress')
    normalized=self%normalized_stress=='true'.or.(self%normalized_stress=='auto'.and..not.self%metric)
    call require(.not.(normalized.and.self%metric),'normalized stress requires nonmetric MDS')
    select case(self%dissimilarity)
    case('euclidean')
      stream=pairwise_distances_chunked(x,working_memory=8._dp*real(n,dp)**2/1048576+1)
      ok=stream%next(chunk);call require(ok,'MDS distance calculation produced no block')
      self%dissimilarity_matrix_=chunk%distance%values
    case('precomputed')
      call require(size(x,2)==n,'MDS dissimilarities must be square')
      call require(all(x>=0),'MDS dissimilarities must be nonnegative')
      call require(all(abs(x-transpose(x))<=1.e-10_dp*max(1._dp,abs(x),abs(transpose(x)))), &
        'MDS dissimilarities must be symmetric')
      do i=1,n
        call require(x(i,i)==0,'MDS dissimilarities must have zero diagonal')
      end do
      self%dissimilarity_matrix_=x
    case default; call require(.false.,'unsupported MDS dissimilarity')
    end select
    if(present(init))then
      call check_x(init);call require(size(init,1)==n,'MDS initial sample count mismatch')
      k=size(init,2)
    end if
    ! Preserve the established native zero-dissimilarity shortcut.
    if(self%metric.and.maxval(self%dissimilarity_matrix_)==0)then
      if(allocated(self%embedding_))deallocate(self%embedding_)
      allocate(self%embedding_(n,k));self%embedding_=0;self%stress_=0
      self%n_iter_=1;self%converged_=.true.;return
    end if
    fit=smacof(self%dissimilarity_matrix_,n_components=self%n_components,metric=self%metric, &
      init=init,n_init=self%n_init,max_iter=self%max_iter,eps=self%eps,random_state=self%random_state, &
      normalized_stress=self%normalized_stress)
    self%embedding_=fit%embedding;self%stress_=fit%stress;self%n_iter_=fit%n_iter;self%converged_=fit%converged
  end subroutine
  function mds_fit_transform(self,x,init) result(z)
    class(mds), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: init(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x,init); z=self%embedding_
  end function
end module
