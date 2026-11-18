! Dense SMACOF following sklearn/manifold/_mds.py, scikit-learn 1.6.1.
! See THIRD_PARTY_NOTICES.md for upstream BSD-3-Clause attribution.
module sklearn_mds
  use sklearn_common
  use sklearn_metrics, only: pairwise_distances
  use sklearn_isotonic, only: isotonic_regression
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
    type(isotonic_regression) :: monotone
    real(dp), allocatable :: position(:,:),distances(:,:),disparities(:,:),b(:,:),ratio(:,:),values(:)
    logical, allocatable :: observed(:,:)
    real(dp) :: stress,old,scaled,norm,denom
    integer :: n,k,runs,run,it,i,j
    logical :: normalized,done
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
    case('euclidean'); self%dissimilarity_matrix_=pairwise_distances(x,x)
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
    runs=self%n_init
    if(present(init)) then
      call check_x(init); call require(size(init,1)==n,'MDS initial sample count mismatch')
      k=size(init,2); runs=1
    end if
    allocate(position(n,k),observed(n,n)); observed=.false.
    do j=1,n
      do i=1,j-1
        observed(i,j)=self%dissimilarity_matrix_(i,j)>0
      end do
    end do
    if(.not.self%metric) call require(any(observed),'nonmetric MDS needs positive dissimilarities')
    if(self%metric.and.maxval(self%dissimilarity_matrix_)==0) then
      position=0; self%embedding_=position; self%stress_=0; self%n_iter_=1; self%converged_=.true.; return
    end if
    call seed_rng(self%random_state); self%stress_=huge(1._dp)
    do run=1,runs
      if(present(init)) then
        position=init
      else
        call random_number(position)
      end if
      old=huge(1._dp); done=.false.
      do it=1,self%max_iter
        distances=pairwise_distances(position,position)
        if(self%metric) then
          disparities=self%dissimilarity_matrix_
        else
          values=monotone%fit_transform(pack(self%dissimilarity_matrix_,observed),pack(distances,observed))
          ! Preserve sklearn 1.6.1's upper-triangle isotonic update and scaling.
          disparities=unpack(values,observed,distances)
          denom=sum(disparities**2)
          call require(denom>tiny(1._dp),'MDS collapsed initial configuration')
          disparities=disparities*sqrt(real(n,dp)*(n-1)/2/denom)
        end if
        stress=sum((distances-disparities)**2)/2
        if(normalized) then
          denom=sum(disparities**2)/2
          call require(denom>tiny(1._dp),'MDS zero disparity norm')
          stress=sqrt(stress/denom)
        end if
        where(distances==0) distances=1.e-5_dp
        ratio=disparities/distances; b=-ratio
        do i=1,n
          b(i,i)=b(i,i)+sum(ratio(i,:))
        end do
        position=matmul(b,position)/n
        norm=sum(sqrt(sum(position**2,dim=2)))
        if(norm<=tiny(1._dp)) then
          call require(stress<=tiny(1._dp),'MDS collapsed initial configuration')
          done=.true.; exit
        end if
        scaled=stress/norm
        if(it>1) then
          if(old-scaled<self%eps) then
            done=.true.; exit
          end if
        end if
        old=scaled
      end do
      if(stress<self%stress_) then
        self%embedding_=position; self%stress_=stress; self%n_iter_=min(it,self%max_iter); self%converged_=done
      end if
    end do
  end subroutine
  function mds_fit_transform(self,x,init) result(z)
    class(mds), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: init(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x,init); z=self%embedding_
  end function
end module
