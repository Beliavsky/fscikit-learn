! BernoulliRBM contracts from sklearn 1.6.1 neural_network/_rbm.py (BSD-3-Clause).
! Upstream notice: LICENSES/scikit-learn-COPYING and THIRD_PARTY_NOTICES.md.
module sklearn_rbm
  use sklearn_common, only: dp,require,check_x
  use sklearn_sparse, only: csr_matrix
  use, intrinsic :: ieee_exceptions
  use, intrinsic :: iso_fortran_env, only: int64
  use sklearn_legacy_rng, only: legacy_random_state
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_fma
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
    procedure, private :: fit_dense=>fit,fit_csr
    generic :: fit=>fit_dense,fit_csr
    procedure, private :: partial_fit_dense=>partial_fit,partial_fit_csr
    generic :: partial_fit=>partial_fit_dense,partial_fit_csr
    procedure, private :: transform_dense=>transform,transform_csr
    generic :: transform=>transform_dense,transform_csr
    procedure, private :: fit_transform_dense=>fit_transform,fit_transform_csr
    generic :: fit_transform=>fit_transform_dense,fit_transform_csr
    procedure, private :: gibbs_dense=>gibbs,gibbs_csr
    generic :: gibbs=>gibbs_dense,gibbs_csr
    procedure, private :: score_samples_dense=>score_samples,score_samples_csr
    generic :: score_samples=>score_samples_dense,score_samples_csr
  end type
contains
  elemental real(dp) function rbm_expit(x) result(value)
    real(dp), intent(in) :: x
    ! scipy.special.xsf/log_exp.h, including negative-logit underflow.
    ! Numerical callers suspend traps and restore the caller's IEEE state.
    value=1/(1+exp(-x))
  end function
  function dense_product(a,b) result(c)
    real(dp), intent(in) :: a(:,:),b(:,:)
    real(dp), allocatable :: c(:,:),products(:),positive(:),negative(:)
    real(dp) :: total
    integer :: i,j,k,np,nn,ip,in
    ! Recover a finite dot product when only its intermediate accumulation
    ! overflowed. Pair opposite signs before consuming a same-sign remainder.
    c=matmul(a,b)
    if(all(ieee_is_finite(c))) return
    allocate(products(size(a,2)),positive(size(a,2)),negative(size(a,2)))
    do j=1,size(b,2)
      do i=1,size(a,1)
        if(ieee_is_finite(c(i,j))) cycle
        products=a(i,:)*b(:,j)
        if(.not.all(ieee_is_finite(products))) then
          ! Keep a finite oversized product fused with an existing infinity,
          ! matching BLAS FMA instead of prematurely forming opposite infinities.
          total=0
          do k=1,size(products)
            total=ieee_fma(a(i,k),b(k,j),total)
          end do
          c(i,j)=total
          cycle
        end if
        np=0; nn=0
        do k=1,size(products)
          if(products(k)>=0) then
            np=np+1; positive(np)=products(k)
          else
            nn=nn+1; negative(nn)=products(k)
          end if
        end do
        total=0; ip=1; in=1
        do while(ip<=np.and.in<=nn)
          if(total>=0) then
            total=total+negative(in); in=in+1
          else
            total=total+positive(ip); ip=ip+1
          end if
        end do
        do while(ip<=np)
          total=total+positive(ip); ip=ip+1
        end do
        do while(in<=nn)
          total=total+negative(in); in=in+1
        end do
        if(ieee_is_finite(total)) c(i,j)=total
      end do
    end do
  end function
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
  subroutine check_sparse(x)
    type(csr_matrix), intent(in) :: x
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'RBM CSR input must have samples and features')
    call require(all(ieee_is_finite(x%data)),'RBM CSR input must be finite')
  end subroutine
  subroutine query_sparse(self,x)
    class(bernoulli_rbm), intent(in) :: self
    type(csr_matrix), intent(in) :: x
    call require(allocated(self%components_),'RBM is not fitted')
    call check_sparse(x)
    call require(x%n_cols==self%n_features_in_,'RBM feature count mismatch')
  end subroutine
  subroutine suspend_ieee(halting,raised)
    logical, intent(out) :: halting(size(ieee_all)),raised(size(ieee_all))
    integer :: i
    do i=1,size(ieee_all)
      call ieee_get_flag(ieee_all(i),raised(i)); call ieee_get_halting_mode(ieee_all(i),halting(i))
      call ieee_set_halting_mode(ieee_all(i),.false.)
    end do
  end subroutine
  subroutine restore_ieee(halting,raised)
    logical, intent(in) :: halting(size(ieee_all)),raised(size(ieee_all))
    integer :: i
    do i=1,size(ieee_all)
      call ieee_set_flag(ieee_all(i),raised(i)); call ieee_set_halting_mode(ieee_all(i),halting(i))
    end do
  end subroutine
  function sparse_rows(x,first,last) result(batch)
    type(csr_matrix), intent(in) :: x
    integer, intent(in) :: first,last
    type(csr_matrix) :: batch
    integer :: start,finish
    start=x%indptr(first); finish=x%indptr(last+1)-1
    batch%n_rows=last-first+1; batch%n_cols=x%n_cols
    batch%indptr=x%indptr(first:last+1)-start+1
    batch%indices=x%indices(start:finish); batch%data=x%data(start:finish)
  end function
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
    logical :: halting(size(ieee_all)),raised(size(ieee_all))
    call suspend_ieee(halting,raised)
    h=rbm_expit(dense_product(x,transpose(self%components_))+spread(self%intercept_hidden_,1,size(x,1)))
    call restore_ieee(halting,raised)
  end function
  function visible_mean(self,h) result(x)
    class(bernoulli_rbm), intent(in) :: self
    real(dp), intent(in) :: h(:,:)
    real(dp) :: x(size(h,1),self%n_features_in_)
    logical :: halting(size(ieee_all)),raised(size(ieee_all))
    call suspend_ieee(halting,raised)
    x=rbm_expit(dense_product(h,self%components_)+spread(self%intercept_visible_,1,size(h,1)))
    call restore_ieee(halting,raised)
  end function
  subroutine binary_sample(probabilities,rng,retain_nan)
    real(dp), intent(inout) :: probabilities(:,:)
    type(legacy_random_state), intent(inout) :: rng
    integer :: i,j
    logical, optional, intent(in) :: retain_nan
    real(dp) :: draw
    do i=1,size(probabilities,1)
      do j=1,size(probabilities,2)
        draw=rng%uniform()
        if(ieee_is_nan(probabilities(i,j))) then
          ! Source persistent-chain floor preserves NaNs; Boolean Gibbs draws do not.
          if(present(retain_nan)) then
            if(retain_nan) cycle
          end if
          probabilities(i,j)=0
        else
          probabilities(i,j)=merge(1._dp,0._dp,draw<probabilities(i,j))
        end if
      end do
    end do
  end subroutine
  subroutine update(self,x,rng,sparse)
    class(bernoulli_rbm), intent(inout) :: self
    real(dp), optional, intent(in) :: x(:,:)
    type(csr_matrix), optional, intent(in) :: sparse
    type(legacy_random_state), intent(inout) :: rng
    real(dp), allocatable :: positive(:,:),negative(:,:),visible(:,:),cross(:,:),totals(:)
    real(dp) :: rate
    integer :: n,i,e,j,k
    logical :: halting(size(ieee_all)),raised(size(ieee_all))
    call suspend_ieee(halting,raised)
    if(present(sparse)) then
      n=sparse%n_rows; positive=hidden_mean_sparse(self,sparse)
    else
      n=size(x,1); positive=hidden_mean(self,x)
    end if
    visible=visible_mean(self,self%h_samples_); call binary_sample(visible,rng)
    negative=hidden_mean(self,visible); rate=self%learning_rate/n
    if(present(sparse)) then
      allocate(cross(size(self%components_,1),self%n_features_in_),totals(self%n_features_in_)); cross=0; totals=0
      do i=1,n
        do e=sparse%indptr(i),sparse%indptr(i+1)-1
          j=sparse%indices(e); totals(j)=totals(j)+sparse%data(e)
          do k=1,size(cross,1)
            cross(k,j)=cross(k,j)+sparse%data(e)*positive(i,k)
          end do
        end do
      end do
    else
      cross=matmul(transpose(positive),x); totals=sum(x,dim=1)
    end if
    self%components_=self%components_+rate*(cross-matmul(transpose(negative),visible))
    self%intercept_hidden_=self%intercept_hidden_+rate*(sum(positive,dim=1)-sum(negative,dim=1))
    self%intercept_visible_=self%intercept_visible_+rate*(totals-sum(visible,dim=1))
    ! Sample from the pre-update conditional, as in the source persistent chain.
    call binary_sample(negative,rng,retain_nan=.true.); self%h_samples_=negative
    call restore_ieee(halting,raised)
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
  function hidden_mean_sparse(self,x) result(h)
    class(bernoulli_rbm), intent(in) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), allocatable :: h(:,:)
    logical :: halting(size(ieee_all)),raised(size(ieee_all))
    call suspend_ieee(halting,raised)
    h=rbm_expit(x%matmul(transpose(self%components_))+spread(self%intercept_hidden_,1,x%n_rows))
    call restore_ieee(halting,raised)
  end function
  function free_energy_sparse(self,x) result(energy)
    class(bernoulli_rbm), intent(in) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), allocatable :: energy(:),visible(:,:),logits(:,:)
    logical :: halting(size(ieee_all)),raised(size(ieee_all))
    call suspend_ieee(halting,raised)
    visible=x%matmul(reshape(self%intercept_visible_,[self%n_features_in_,1]))
    logits=x%matmul(transpose(self%components_))+spread(self%intercept_hidden_,1,x%n_rows)
    energy=-visible(:,1)-sum(softplus(logits),dim=2)
    call restore_ieee(halting,raised)
  end function
  function corrupt_sparse(x,columns) result(z)
    type(csr_matrix), intent(in) :: x
    integer, intent(in) :: columns(:)
    type(csr_matrix) :: z
    real(dp), allocatable :: row(:)
    integer, allocatable :: links(:)
    real(dp) :: value,delta,selected
    integer :: i,j,e,position,first,last,head,next,column
    logical :: canonical,inserted
    call require(int(x%nnz(),int64)+x%n_rows<int(huge(1),int64),'RBM corrupted CSR stored count overflow')
    canonical=.true.
    do i=1,x%n_rows
      first=x%indptr(i); last=x%indptr(i+1)-1
      if(last>first) then
        if(any(x%indices(first+1:last)<=x%indices(first:last-1))) canonical=.false.
      end if
    end do
    z%n_rows=x%n_rows; z%n_cols=x%n_cols
    allocate(z%indptr(x%n_rows+1),z%indices(x%nnz()+x%n_rows),z%data(x%nnz()+x%n_rows)); position=1
    if(.not.canonical) then
      allocate(row(x%n_cols),links(x%n_cols)); row=0; links=0
    end if
    do i=1,x%n_rows
      z%indptr(i)=position; column=columns(i); selected=0
      first=x%indptr(i); last=x%indptr(i+1)-1
      do e=first,last
        if(x%indices(e)==column) selected=selected+x%data(e)
      end do
      delta=-2*selected+1
      if(canonical) then
        ! SciPy's canonical CSR addition merges sorted A and the one-entry B.
        e=first; inserted=.false.
        do while(e<=last.or..not.inserted)
          if(e>last) then
            j=column; value=0+delta; inserted=.true.
          else if(.not.inserted.and.x%indices(e)>column) then
            j=column; value=0+delta; inserted=.true.
          else
            j=x%indices(e); value=x%data(e)+0
            if(j==column) then
              value=x%data(e)+delta; inserted=.true.
            end if
            e=e+1
          end if
          if(value==0) cycle
          z%indices(position)=j; z%data(position)=value; position=position+1
        end do
      else
        ! SciPy's general CSR addition sums each A row, then B, and visits
        ! touched columns in reverse first-touch order through a linked list.
        head=-1
        do e=first,last
          j=x%indices(e); row(j)=row(j)+x%data(e)
          if(links(j)==0) then
            links(j)=head; head=j
          end if
        end do
        if(links(column)==0) then
          links(column)=head; head=column
        end if
        do while(head>0)
          value=row(head)+0
          if(head==column) value=row(head)+delta
          if(value/=0) then
            z%indices(position)=head; z%data(position)=value; position=position+1
          end if
          next=links(head); links(head)=0; row(head)=0; head=next
        end do
      end if
    end do
    z%indptr(x%n_rows+1)=position
    z%indices=z%indices(:position-1); z%data=z%data(:position-1)
  end function
  function score_samples_csr(self,x) result(scores)
    class(bernoulli_rbm), intent(in) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), allocatable :: scores(:)
    type(csr_matrix) :: corrupted
    type(legacy_random_state) :: rng
    integer :: i,columns(x%n_rows)
    logical :: halting(size(ieee_all)),raised(size(ieee_all))
    call query_sparse(self,x)
    call require(self%random_state>=0,'RBM random_state must be nonnegative')
    call suspend_ieee(halting,raised); call rng%seed(self%random_state)
    do i=1,x%n_rows
      columns(i)=rng%integer_index(x%n_cols)
    end do
    corrupted=corrupt_sparse(x,columns)
    scores=-real(x%n_cols,dp)*softplus(free_energy_sparse(self,x)-free_energy_sparse(self,corrupted))
    call restore_ieee(halting,raised)
  end function
  function free_energy(self,x) result(energy)
    class(bernoulli_rbm), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: energy(size(x,1))
    real(dp), allocatable :: visible(:,:),logits(:,:)
    logical :: halting(size(ieee_all)),raised(size(ieee_all))
    call suspend_ieee(halting,raised)
    visible=dense_product(x,reshape(self%intercept_visible_,[self%n_features_in_,1]))
    logits=dense_product(x,transpose(self%components_))+spread(self%intercept_hidden_,1,size(x,1))
    energy=-visible(:,1)-sum(softplus(logits),dim=2)
    call restore_ieee(halting,raised)
  end function
  function score_samples(self,x) result(scores)
    class(bernoulli_rbm), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: scores(:),corrupted(:,:)
    type(legacy_random_state) :: rng
    integer :: i,j
    logical :: halting(size(ieee_all)),raised(size(ieee_all))
    call query(self,x)
    call require(self%random_state>=0,'RBM random_state must be nonnegative')
    call suspend_ieee(halting,raised)
    call rng%seed(self%random_state); corrupted=x
    do i=1,size(x,1)
      j=rng%integer_index(size(x,2)); corrupted(i,j)=1-corrupted(i,j)
    end do
    scores=-real(size(x,2),dp)*softplus(free_energy(self,x)-free_energy(self,corrupted))
    call restore_ieee(halting,raised)
  end function
  subroutine fit_csr(self,x)
    class(bernoulli_rbm), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    type(legacy_random_state) :: rng
    integer :: iteration,first,last
    call controls(self); call check_sparse(x)
    call rng%seed(self%random_state); call initialize(self,x%n_cols,rng)
    do iteration=1,self%n_iter
      first=1
      do
        last=first+min(self%batch_size,x%n_rows-first+1)-1
        call update(self,rng=rng,sparse=sparse_rows(x,first,last))
        if(last==x%n_rows) exit
        first=last+1
      end do
    end do
    ! fit's RNG is local; an existing Gibbs/partial_fit stream is preserved.
  end subroutine
  subroutine partial_fit_csr(self,x)
    class(bernoulli_rbm), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    type(legacy_random_state) :: rng
    call controls(self); call check_sparse(x)
    if(.not.self%sampling_initialized) then
      call self%sampling_rng%seed(self%random_state); self%sampling_initialized=.true.
    end if
    ! Work on a copy to avoid aliasing self and its RNG in the update call.
    rng=self%sampling_rng
    if(.not.allocated(self%components_)) then
      call initialize(self,x%n_cols,rng)
    else
      call query_sparse(self,x)
      call require(self%n_components==size(self%components_,1),'RBM n_components changed; use fit to reset')
    end if
    call update(self,rng=rng,sparse=x); self%sampling_rng=rng
  end subroutine
  function transform_csr(self,x) result(h)
    class(bernoulli_rbm), intent(in) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), allocatable :: h(:,:)
    call query_sparse(self,x); h=hidden_mean_sparse(self,x)
  end function
  function fit_transform_csr(self,x) result(h)
    class(bernoulli_rbm), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), allocatable :: h(:,:)
    call self%fit(x); h=self%transform(x)
  end function
  function gibbs_csr(self,x) result(sample)
    class(bernoulli_rbm), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), allocatable :: sample(:,:),h(:,:)
    call query_sparse(self,x)
    if(.not.self%sampling_initialized) then
      call require(self%random_state>=0,'RBM random_state must be nonnegative')
      call self%sampling_rng%seed(self%random_state); self%sampling_initialized=.true.
    end if
    h=hidden_mean_sparse(self,x); call binary_sample(h,self%sampling_rng)
    sample=visible_mean(self,h); call binary_sample(sample,self%sampling_rng)
  end function
end module
