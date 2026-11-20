! Dense single-worker sklearn 1.6.1 decomposition/_lda.py and _online_lda_fast.pyx contracts (BSD).
! NumPy 2.2.6 legacy_standard_gamma shape-100 branch; notices in LICENSES/numpy-*.
! See THIRD_PARTY_NOTICES.md and LICENSES/scikit-learn-COPYING.
module sklearn_lda
  use sklearn_common, only: dp,require,check_x,logsumexp
  use sklearn_legacy_rng, only: legacy_random_state
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_positive_inf,ieee_quiet_nan
  implicit none
  private
  public :: latent_dirichlet_allocation
  type :: latent_dirichlet_allocation
    integer :: n_components=10,max_iter=10,batch_size=128,evaluate_every=-1,max_doc_update_iter=100,random_state=0
    integer :: n_features_in_=0,n_iter_=0
    integer(int64) :: n_batch_iter_=1
    real(dp) :: doc_topic_prior=-1,topic_word_prior=-1,learning_decay=.7_dp,learning_offset=10
    real(dp) :: total_samples=1.e6_dp,perp_tol=.1_dp,mean_change_tol=.001_dp
    real(dp) :: doc_topic_prior_=0,topic_word_prior_=0,bound_=0
    character(8) :: learning_method='batch'
    real(dp), allocatable :: components_(:,:),exp_dirichlet_component_(:,:)
    type(legacy_random_state), private :: rng
  contains
    procedure :: fit
    procedure :: partial_fit
    procedure :: transform
    procedure :: fit_transform
    procedure :: score
    procedure :: perplexity
  end type
contains
  subroutine controls(self)
    class(latent_dirichlet_allocation), intent(in) :: self
    call require(self%n_components>=1,'LDA n_components must be positive')
    call require(self%batch_size>=1,'LDA batch_size must be positive')
    call require(self%max_iter>=0.and.self%max_doc_update_iter>=0,'LDA iteration limits must be nonnegative')
    call require(self%random_state>=0,'LDA random_state must be nonnegative')
    call require(self%learning_method=='batch'.or.self%learning_method=='online','unknown LDA learning_method')
    call require(ieee_is_finite(self%doc_topic_prior).and.ieee_is_finite(self%topic_word_prior).and. &
      ieee_is_finite(self%learning_decay).and.ieee_is_finite(self%learning_offset).and. &
      ieee_is_finite(self%total_samples).and.ieee_is_finite(self%perp_tol).and. &
      ieee_is_finite(self%mean_change_tol),'LDA numeric controls must be finite')
    call require(self%doc_topic_prior==-1.or.(self%doc_topic_prior>0.and.self%doc_topic_prior<=1), &
      'LDA doc_topic_prior must be in (0,1] or -1 for auto')
    call require(self%topic_word_prior==-1.or.(self%topic_word_prior>0.and.self%topic_word_prior<=1), &
      'LDA topic_word_prior must be in (0,1] or -1 for auto')
    call require(self%learning_decay>=0.and.self%learning_decay<=1,'LDA learning_decay must be in [0,1]')
    call require(self%learning_offset>=1,'LDA learning_offset must be at least one')
    call require(self%total_samples>0,'LDA total_samples must be positive')
    call require(self%perp_tol>=0.and.self%mean_change_tol>=0,'LDA tolerances must be nonnegative')
  end subroutine
  subroutine data_check(x)
    real(dp), intent(in) :: x(:,:)
    call check_x(x)
    call require(all(x>=0),'LDA X must be nonnegative')
  end subroutine
  subroutine query(self,x)
    class(latent_dirichlet_allocation), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    call require(allocated(self%components_),'LDA is not fitted')
    call data_check(x)
    call require(size(x,2)==self%n_features_in_,'LDA feature count mismatch')
    call require(self%n_components==size(self%components_,1),'LDA n_components changed; use fit to reset')
  end subroutine
  elemental real(dp) function psi(x) result(value)
    real(dp), intent(in) :: x
    real(dp) :: t,r
    ! Source AS 103 approximation, intentionally matching its speed/accuracy tradeoff.
    if(x<=1.e-6_dp) then
      value=-.577215664901532860606512090082402431_dp-1/x; return
    end if
    t=x; value=0
    do while(t<6)
      value=value-1/t; t=t+1
    end do
    r=1/t; value=value+log(t)-.5_dp*r; r=r*r
    value=value-r*(1._dp/12-r*(1._dp/120-r/252))
  end function
  function expectation(values) result(log_mean)
    real(dp), intent(in) :: values(:)
    real(dp) :: log_mean(size(values))
    log_mean=psi(values)-psi(sum(values))
  end function
  function exp_expectation(values) result(exp_mean)
    real(dp), intent(in) :: values(:,:)
    real(dp) :: exp_mean(size(values,1),size(values,2))
    integer :: i
    do i=1,size(values,1)
      exp_mean(i,:)=exp(expectation(values(i,:)))
    end do
  end function
  real(dp) function gamma100(rng) result(value)
    type(legacy_random_state), intent(inout) :: rng
    real(dp), parameter :: b=100._dp-1._dp/3,c=1/sqrt(9*b)
    real(dp) :: x,v,u
    do
      do
        x=rng%normal(); v=1+c*x
        if(v>0) exit
      end do
      v=v*v*v; u=rng%uniform()
      if(u<1-.0331_dp*(x*x)*(x*x)) exit
      ! log(U) tends to negative infinity at zero; accept without raising a trap.
      if(u==0) exit
      if(log(u)<.5_dp*x*x+b*(1-v+log(v))) exit
    end do
    value=.01_dp*(b*v)
  end function
  subroutine initialize(self,p)
    class(latent_dirichlet_allocation), intent(inout) :: self
    integer, intent(in) :: p
    integer :: i,j
    call self%rng%seed(self%random_state)
    self%n_features_in_=p; self%n_iter_=0; self%n_batch_iter_=1
    self%bound_=ieee_value(0._dp,ieee_quiet_nan)
    self%doc_topic_prior_=self%doc_topic_prior
    if(self%doc_topic_prior==-1) self%doc_topic_prior_=1._dp/self%n_components
    self%topic_word_prior_=self%topic_word_prior
    if(self%topic_word_prior==-1) self%topic_word_prior_=1._dp/self%n_components
    self%components_=spread(spread(0._dp,1,self%n_components),2,p)
    do i=1,self%n_components
      do j=1,p
        self%components_(i,j)=gamma100(self%rng)
      end do
    end do
    self%exp_dirichlet_component_=exp_expectation(self%components_)
  end subroutine
  subroutine e_step(self,x,docs,statistics,rng)
    class(latent_dirichlet_allocation), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable, intent(out) :: docs(:,:)
    real(dp), allocatable, optional, intent(out) :: statistics(:,:)
    type(legacy_random_state), optional, intent(inout) :: rng
    integer, allocatable :: ids(:)
    real(dp), allocatable :: word_topics(:,:),counts(:),norm(:)
    real(dp) :: exp_doc(size(self%components_,1)),old(size(self%components_,1))
    integer :: i,j,k,iteration,n,p,c
    n=size(x,1); p=size(x,2); c=size(self%components_,1)
    allocate(docs(n,c)); docs=1
    if(present(rng)) then
      do i=1,n
        do k=1,c
          docs(i,k)=gamma100(rng)
        end do
      end do
    end if
    if(present(statistics)) then
      allocate(statistics(c,p)); statistics=0
    end if
    do i=1,n
      ids=pack([(j,j=1,p)],x(i,:)/=0); counts=x(i,ids); word_topics=self%exp_dirichlet_component_(:,ids)
      exp_doc=exp(expectation(docs(i,:)))
      do iteration=1,self%max_doc_update_iter
        old=docs(i,:); norm=matmul(exp_doc,word_topics)+epsilon(1._dp)
        docs(i,:)=exp_doc*matmul(word_topics,counts/norm)+self%doc_topic_prior_
        exp_doc=exp(expectation(docs(i,:)))
        if(sum(abs(old-docs(i,:)))/c<self%mean_change_tol) exit
      end do
      if(present(statistics)) then
        norm=matmul(exp_doc,word_topics)+epsilon(1._dp)
        do j=1,size(ids)
          statistics(:,ids(j))=statistics(:,ids(j))+exp_doc*(counts(j)/norm(j))
        end do
      end if
    end do
    if(present(statistics)) statistics=statistics*self%exp_dirichlet_component_
  end subroutine
  subroutine em_step(self,x,total,batch_update)
    class(latent_dirichlet_allocation), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),total
    logical, intent(in) :: batch_update
    real(dp), allocatable :: docs(:,:),statistics(:,:)
    real(dp) :: weight,ratio
    type(legacy_random_state) :: rng
    rng=self%rng; call e_step(self,x,docs,statistics,rng); self%rng=rng
    if(batch_update) then
      self%components_=self%topic_word_prior_+statistics
    else
      weight=(self%learning_offset+real(self%n_batch_iter_,dp))**(-self%learning_decay)
      ratio=total/size(x,1)
      self%components_=(1-weight)*self%components_+weight*(self%topic_word_prior_+ratio*statistics)
    end if
    self%exp_dirichlet_component_=exp_expectation(self%components_)
    call require(self%n_batch_iter_<huge(self%n_batch_iter_),'LDA batch counter capacity exceeded')
    self%n_batch_iter_=self%n_batch_iter_+1
  end subroutine
  subroutine chunks(self,x,total)
    class(latent_dirichlet_allocation), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),total
    integer :: first,last
    first=1
    do
      last=first+min(self%batch_size,size(x,1)-first+1)-1
      call em_step(self,x(first:last,:),total,.false.)
      if(last==size(x,1)) exit
      first=last+1
    end do
  end subroutine
  subroutine fit(self,x)
    class(latent_dirichlet_allocation), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: current,previous
    logical :: evaluated
    integer :: i
    call controls(self); call data_check(x); call initialize(self,size(x,2))
    evaluated=.false.; previous=0
    do i=1,self%max_iter
      if(self%learning_method=='online') then
        call chunks(self,x,real(size(x,1),dp))
      else
        call em_step(self,x,real(size(x,1),dp),.true.)
      end if
      if(self%evaluate_every>0) then
        if(mod(i,self%evaluate_every)==0) then
          current=self%perplexity(x)
          if(evaluated) then
            if(ieee_is_finite(previous).and.ieee_is_finite(current)) then
              if(previous/=0) then
                if(abs(previous-current)<self%perp_tol) exit
              end if
            end if
          end if
          previous=current; evaluated=.true.
        end if
      end if
      ! Source does not increment n_iter_ for the iteration triggering early stopping.
      self%n_iter_=self%n_iter_+1
    end do
    self%bound_=self%perplexity(x)
  end subroutine
  subroutine partial_fit(self,x)
    class(latent_dirichlet_allocation), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    call controls(self); call data_check(x)
    if(.not.allocated(self%components_)) then
      call initialize(self,size(x,2))
    else
      call query(self,x)
    end if
    call chunks(self,x,self%total_samples)
  end subroutine
  function transform(self,x,normalize) result(docs)
    class(latent_dirichlet_allocation), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    logical, optional, intent(in) :: normalize
    real(dp), allocatable :: docs(:,:)
    logical :: normalized
    integer :: i
    call query(self,x); call e_step(self,x,docs)
    normalized=.true.
    if(present(normalize)) normalized=normalize
    if(normalized) then
      do i=1,size(docs,1)
        docs(i,:)=docs(i,:)/sum(docs(i,:))
      end do
    end if
  end function
  function fit_transform(self,x,normalize) result(docs)
    class(latent_dirichlet_allocation), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    logical, optional, intent(in) :: normalize
    real(dp), allocatable :: docs(:,:)
    call self%fit(x); docs=self%transform(x,normalize)
  end function
  elemental real(dp) function gamma_psi_remainder(x) result(value)
    real(dp), intent(in) :: x
    real(dp) :: r,r2
    ! logGamma(x)-x*psi(x)+x, with source psi and a stable Stirling remainder.
    if(x<16) then
      value=log_gamma(x)-x*psi(x)+x
    else
      r=1/x; r2=r*r
      value=.5_dp*(1+log(2*acos(-1._dp))-log(x))+r*(1._dp/6+r2*(-1._dp/90+ &
        r2*(1._dp/210+r2*(-1._dp/1680+r2*(1._dp/1188+r2*(-691._dp/360360+r2/156))))))
    end if
  end function
  real(dp) function dirichlet_bound(prior,values,log_mean,dimension) result(value)
    real(dp), intent(in) :: prior,values(:,:),log_mean(:,:)
    integer, intent(in) :: dimension
    integer :: i
    ! Algebraically eliminate the O(sum(values)) terms before summing.
    value=0
    if(dimension==1) return
    do i=1,size(values,1)
      if(all(abs(values(i,:)-prior)<=8*epsilon(1._dp)*prior)) cycle
      value=value+prior*sum(log_mean(i,:))+log_gamma(prior*dimension)-dimension*log_gamma(prior)+ &
        sum(gamma_psi_remainder(values(i,:)))-gamma_psi_remainder(sum(values(i,:)))
    end do
  end function
  real(dp) function approx_bound(self,x,docs,sub_sampling) result(value)
    class(latent_dirichlet_allocation), intent(in) :: self
    real(dp), intent(in) :: x(:,:),docs(:,:)
    logical, intent(in) :: sub_sampling
    real(dp) :: log_docs(size(docs,1),size(docs,2)),log_topics(size(self%components_,1),size(x,2))
    integer :: i,j
    do i=1,size(docs,1)
      log_docs(i,:)=expectation(docs(i,:))
    end do
    do i=1,size(self%components_,1)
      log_topics(i,:)=expectation(self%components_(i,:))
    end do
    value=0
    do i=1,size(x,1)
      do j=1,size(x,2)
        if(x(i,j)==0) cycle
        value=value+x(i,j)*logsumexp(log_docs(i,:)+log_topics(:,j))
      end do
    end do
    value=value+dirichlet_bound(self%doc_topic_prior_,docs,log_docs,self%n_components)
    if(sub_sampling) value=value*(self%total_samples/size(x,1))
    value=value+dirichlet_bound(self%topic_word_prior_,self%components_,log_topics,size(x,2))
  end function
  real(dp) function score(self,x) result(value)
    class(latent_dirichlet_allocation), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: docs(:,:)
    call query(self,x); call e_step(self,x,docs)
    value=approx_bound(self,x,docs,.false.)
  end function
  real(dp) function perplexity(self,x,sub_sampling) result(value)
    class(latent_dirichlet_allocation), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    logical, optional, intent(in) :: sub_sampling
    real(dp), allocatable :: docs(:,:)
    real(dp) :: bound,words,exponent
    logical :: subsample
    call query(self,x); subsample=.false.
    if(present(sub_sampling)) subsample=sub_sampling
    call e_step(self,x,docs); bound=approx_bound(self,x,docs,subsample); words=sum(x)
    if(subsample) words=words*(self%total_samples/size(x,1))
    if(words==0) then
      if(bound<0) then
        value=ieee_value(0._dp,ieee_positive_inf)
      else if(bound>0) then
        value=0
      else
        value=ieee_value(0._dp,ieee_quiet_nan)
      end if
      return
    end if
    exponent=-bound/words
    if(exponent>log(huge(1._dp))) then
      value=ieee_value(0._dp,ieee_positive_inf)
    else
      value=exp(exponent)
    end if
  end function
end module
