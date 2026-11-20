! BernoulliRBM contracts from sklearn 1.6.1 neural_network/_rbm.py (BSD-3-Clause).
! Upstream notice: LICENSES/scikit-learn-COPYING and THIRD_PARTY_NOTICES.md.
module sklearn_rbm
  use sklearn_common, only: dp,require,check_x,sigmoid
  use sklearn_legacy_rng, only: legacy_random_state
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: bernoulli_rbm
  type :: bernoulli_rbm
    integer :: n_components=256,batch_size=10,n_iter=10,random_state=0,n_features_in_=0
    real(dp) :: learning_rate=.1_dp
    real(dp), allocatable :: components_(:,:),intercept_hidden_(:),intercept_visible_(:),h_samples_(:,:)
    type(legacy_random_state), private :: sampling_rng
    logical, private :: sampling_initialized=.false.
  contains
    procedure :: fit
    procedure :: partial_fit
    procedure :: transform
    procedure :: fit_transform
    procedure :: gibbs
    procedure :: score_samples
  end type
contains
  subroutine controls(self)
    class(bernoulli_rbm), intent(in) :: self
    call require(self%n_components>=1,'RBM n_components must be positive')
    call require(self%batch_size>=1,'RBM batch_size must be positive')
    call require(self%n_iter>=0,'RBM n_iter must be nonnegative')
    call require(ieee_is_finite(self%learning_rate),'RBM learning_rate must be finite')
    call require(self%learning_rate>0,'RBM learning_rate must be positive')
    call require(self%random_state>=0,'RBM random_state must be nonnegative')
  end subroutine
  subroutine query(self,x)
    class(bernoulli_rbm), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    call require(allocated(self%components_),'RBM is not fitted')
    call check_x(x)
    call require(size(x,2)==self%n_features_in_,'RBM feature count mismatch')
  end subroutine
  subroutine initialize(self,p,rng)
    class(bernoulli_rbm), intent(inout) :: self
    integer, intent(in) :: p
    type(legacy_random_state), intent(inout) :: rng
    integer :: i,j
    self%n_features_in_=p
    self%components_=spread(spread(0._dp,1,self%n_components),2,p)
    ! NumPy draws in logical row-major order before storing its Fortran array.
    do i=1,self%n_components
      do j=1,p
        self%components_(i,j)=.01_dp*rng%normal()
      end do
    end do
    self%intercept_hidden_=spread(0._dp,1,self%n_components)
    self%intercept_visible_=spread(0._dp,1,p)
    self%h_samples_=spread(spread(0._dp,1,self%batch_size),2,self%n_components)
  end subroutine
  function hidden_mean(self,x) result(h)
    class(bernoulli_rbm), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: h(size(x,1),size(self%components_,1))
    h=sigmoid(matmul(x,transpose(self%components_))+spread(self%intercept_hidden_,1,size(x,1)))
  end function
  function visible_mean(self,h) result(x)
    class(bernoulli_rbm), intent(in) :: self
    real(dp), intent(in) :: h(:,:)
    real(dp) :: x(size(h,1),self%n_features_in_)
    x=sigmoid(matmul(h,self%components_)+spread(self%intercept_visible_,1,size(h,1)))
  end function
  subroutine binary_sample(probabilities,rng)
    real(dp), intent(inout) :: probabilities(:,:)
    type(legacy_random_state), intent(inout) :: rng
    integer :: i,j
    do i=1,size(probabilities,1)
      do j=1,size(probabilities,2)
        probabilities(i,j)=merge(1._dp,0._dp,rng%uniform()<probabilities(i,j))
      end do
    end do
  end subroutine
  subroutine update(self,x,rng)
    class(bernoulli_rbm), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(legacy_random_state), intent(inout) :: rng
    real(dp), allocatable :: positive(:,:),negative(:,:),visible(:,:)
    real(dp) :: rate
    positive=hidden_mean(self,x)
    visible=visible_mean(self,self%h_samples_); call binary_sample(visible,rng)
    negative=hidden_mean(self,visible); rate=self%learning_rate/size(x,1)
    self%components_=self%components_+rate*(matmul(transpose(positive),x)-matmul(transpose(negative),visible))
    self%intercept_hidden_=self%intercept_hidden_+rate*(sum(positive,dim=1)-sum(negative,dim=1))
    self%intercept_visible_=self%intercept_visible_+rate*(sum(x,dim=1)-sum(visible,dim=1))
    ! Sample from the pre-update conditional, as in the source persistent chain.
    call binary_sample(negative,rng); self%h_samples_=negative
  end subroutine
  subroutine fit(self,x)
    class(bernoulli_rbm), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(legacy_random_state) :: rng
    integer :: iteration,first,last
    call controls(self); call check_x(x)
    call rng%seed(self%random_state); call initialize(self,size(x,2),rng)
    do iteration=1,self%n_iter
      first=1
      do
        last=first+min(self%batch_size,size(x,1)-first+1)-1
        call update(self,x(first:last,:),rng)
        if(last==size(x,1)) exit
        first=last+1
      end do
    end do
    ! fit's RNG is local; an existing Gibbs/partial_fit stream is preserved.
  end subroutine
  subroutine partial_fit(self,x)
    class(bernoulli_rbm), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(legacy_random_state) :: rng
    call controls(self); call check_x(x)
    if(.not.self%sampling_initialized) then
      call self%sampling_rng%seed(self%random_state); self%sampling_initialized=.true.
    end if
    ! Work on a copy to avoid aliasing self and its RNG in the update call.
    rng=self%sampling_rng
    if(.not.allocated(self%components_)) then
      call initialize(self,size(x,2),rng)
    else
      call query(self,x)
      call require(self%n_components==size(self%components_,1),'RBM n_components changed; use fit to reset')
    end if
    call update(self,x,rng); self%sampling_rng=rng
  end subroutine
  function transform(self,x) result(h)
    class(bernoulli_rbm), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: h(:,:)
    call query(self,x); h=hidden_mean(self,x)
  end function
  function fit_transform(self,x) result(h)
    class(bernoulli_rbm), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: h(:,:)
    call self%fit(x); h=self%transform(x)
  end function
  function gibbs(self,x) result(sample)
    class(bernoulli_rbm), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: sample(:,:),h(:,:)
    call query(self,x)
    if(.not.self%sampling_initialized) then
      call require(self%random_state>=0,'RBM random_state must be nonnegative')
      call self%sampling_rng%seed(self%random_state); self%sampling_initialized=.true.
    end if
    h=hidden_mean(self,x); call binary_sample(h,self%sampling_rng)
    sample=visible_mean(self,h); call binary_sample(sample,self%sampling_rng)
  end function
  elemental real(dp) function softplus(x) result(value)
    real(dp), intent(in) :: x
    real(dp) :: tail,one_plus
    tail=exp(-abs(x)); one_plus=1+tail
    if(one_plus==1) then
      value=max(0._dp,x)+tail
    else
      value=max(0._dp,x)+log(one_plus)*tail/(one_plus-1)
    end if
  end function
  function free_energy(self,x) result(energy)
    class(bernoulli_rbm), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: energy(size(x,1))
    energy=-matmul(x,self%intercept_visible_)-sum(softplus( &
      matmul(x,transpose(self%components_))+spread(self%intercept_hidden_,1,size(x,1))),dim=2)
  end function
  function score_samples(self,x) result(scores)
    class(bernoulli_rbm), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: scores(:),corrupted(:,:)
    type(legacy_random_state) :: rng
    integer :: i,j
    call query(self,x)
    call require(self%random_state>=0,'RBM random_state must be nonnegative')
    call rng%seed(self%random_state); corrupted=x
    do i=1,size(x,1)
      j=rng%integer_index(size(x,2)); corrupted(i,j)=1-corrupted(i,j)
    end do
    scores=-real(size(x,2),dp)*softplus(free_energy(self,x)-free_energy(self,corrupted))
  end function
end module
