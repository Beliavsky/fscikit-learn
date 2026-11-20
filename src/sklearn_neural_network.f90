module sklearn_neural_network
  use sklearn_rbm, only: bernoulli_rbm
  use sklearn_common
  use sklearn_metrics, only: accuracy_score,r2_score
  implicit none
  private
  public :: mlp_classifier,mlp_regressor,bernoulli_rbm
  type :: dense_layer
    real(dp), allocatable :: weights(:,:),bias(:),m(:,:),v(:,:),mb(:),vb(:)
  end type
  type :: activation_buffer
    real(dp), allocatable :: values(:,:),delta(:,:)
  end type
  type :: network_storage
    integer, allocatable :: hidden_layer_sizes(:)
    integer :: max_iter=1000,random_state=0,n_iter_=0,n_iter_no_change=20
    real(dp) :: learning_rate_init=0.001_dp,alpha=0.0001_dp,tol=1.e-6_dp,loss_=0
    character(16) :: activation='relu'
    logical :: converged_=.false.
    type(dense_layer), allocatable :: layers_(:)
    real(dp), allocatable :: loss_curve_(:)
  end type
  type, extends(network_storage) :: mlp_classifier
    integer, allocatable :: classes_(:)
  contains
    procedure :: fit=>mlpc_fit
    procedure :: predict_proba=>mlpc_proba
    procedure :: predict=>mlpc_predict
    procedure :: score=>mlpc_score
  end type
  type, extends(network_storage) :: mlp_regressor
  contains
    procedure :: fit=>mlpr_fit
    procedure :: predict=>mlpr_predict
    procedure :: score=>mlpr_score
  end type
contains
  function activate(x,kind) result(z)
    real(dp), intent(in) :: x(:,:)
    character(*), intent(in) :: kind
    real(dp) :: z(size(x,1),size(x,2))
    select case(trim(kind))
    case('relu'); z=max(0._dp,x)
    case('tanh'); z=tanh(x)
    case('logistic'); z=sigmoid(x)
    case('identity'); z=x
    case default; call require(.false.,'unknown MLP activation')
    end select
  end function
  function derivative(activated,kind) result(z)
    real(dp), intent(in) :: activated(:,:)
    character(*), intent(in) :: kind
    real(dp) :: z(size(activated,1),size(activated,2))
    select case(trim(kind))
    case('relu'); z=merge(1._dp,0._dp,activated>0)
    case('tanh'); z=1-activated**2
    case('logistic'); z=activated*(1-activated)
    case('identity'); z=1
    end select
  end function
  subroutine initialize(self,p,outputs)
    class(network_storage), intent(inout) :: self
    integer, intent(in) :: p,outputs
    integer, allocatable :: sizes(:)
    integer :: l,inputs,width
    real(dp) :: bound
    call require(self%max_iter>0.and.self%learning_rate_init>0.and.self%alpha>=0.and.self%tol>0, &
      'invalid MLP training controls')
    call require(self%n_iter_no_change>0,'MLP n_iter_no_change must be positive')
    call require(any(self%activation==['relu    ','tanh    ','logistic','identity']),'unknown MLP activation')
    if(.not.allocated(self%hidden_layer_sizes)) self%hidden_layer_sizes=[100]
    call require(all(self%hidden_layer_sizes>0),'hidden layer sizes must be positive')
    sizes=[p,self%hidden_layer_sizes,outputs]
    if(allocated(self%layers_)) deallocate(self%layers_)
    allocate(self%layers_(size(sizes)-1)); call seed_rng(self%random_state)
    do l=1,size(self%layers_)
      inputs=sizes(l); width=sizes(l+1); bound=sqrt(6._dp/(inputs+width))
      if(self%activation=='logistic') bound=sqrt(2._dp/(inputs+width))
      allocate(self%layers_(l)%weights(inputs,width),self%layers_(l)%bias(width), &
        self%layers_(l)%m(inputs,width),self%layers_(l)%v(inputs,width), &
        self%layers_(l)%mb(width),self%layers_(l)%vb(width))
      call random_number(self%layers_(l)%weights); call random_number(self%layers_(l)%bias)
      self%layers_(l)%weights=(2*self%layers_(l)%weights-1)*bound
      self%layers_(l)%bias=(2*self%layers_(l)%bias-1)*bound
      self%layers_(l)%m=0; self%layers_(l)%v=0; self%layers_(l)%mb=0; self%layers_(l)%vb=0
    end do
  end subroutine
  subroutine forward(self,x,classification,buffers)
    class(network_storage), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    logical, intent(in) :: classification
    type(activation_buffer), allocatable, intent(out) :: buffers(:)
    integer :: l,i,last
    call require(allocated(self%layers_),'MLP is not fitted')
    call check_xy(transpose(x),size(self%layers_(1)%weights,1))
    last=size(self%layers_); allocate(buffers(0:last)); buffers(0)%values=x
    do l=1,last
      buffers(l)%values=matmul(buffers(l-1)%values,self%layers_(l)%weights)+ &
        spread(self%layers_(l)%bias,1,size(x,1))
      if(l<last) then
        buffers(l)%values=activate(buffers(l)%values,self%activation)
      else if(classification) then
        do i=1,size(x,1)
          buffers(l)%values(i,:)=exp(buffers(l)%values(i,:)-logsumexp(buffers(l)%values(i,:)))
        end do
      end if
    end do
  end subroutine
  subroutine train(self,x,target,classification)
    class(network_storage), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),target(:,:)
    logical, intent(in) :: classification
    type(activation_buffer), allocatable :: buffers(:)
    real(dp), allocatable :: gw(:,:),gb(:)
    real(dp) :: loss,best,rate,penalty
    integer :: it,l,last,n,stale
    call check_xy(x,size(target,1)); call check_x(target)
    n=size(x,1); call initialize(self,size(x,2),size(target,2)); last=size(self%layers_)
    self%loss_curve_=spread(0._dp,1,self%max_iter); best=huge(1._dp); stale=0; self%converged_=.false.
    do it=1,self%max_iter
      call forward(self,x,classification,buffers); penalty=0
      do l=1,last
        penalty=penalty+sum(self%layers_(l)%weights**2)
      end do
      if(classification) then
        loss=-sum(target*log(max(buffers(last)%values,tiny(1._dp))))/n
      else
        loss=sum((buffers(last)%values-target)**2)/(2*n)
      end if
      loss=loss+self%alpha*penalty/(2*n); self%loss_curve_(it)=loss
      buffers(last)%delta=buffers(last)%values-target
      do l=last-1,1,-1
        buffers(l)%delta=matmul(buffers(l+1)%delta,transpose(self%layers_(l+1)%weights))* &
          derivative(buffers(l)%values,self%activation)
      end do
      rate=self%learning_rate_init*sqrt(1-0.999_dp**it)/(1-0.9_dp**it)
      do l=1,last
        gw=(matmul(transpose(buffers(l-1)%values),buffers(l)%delta)+self%alpha*self%layers_(l)%weights)/n
        gb=sum(buffers(l)%delta,dim=1)/n
        self%layers_(l)%m=0.9_dp*self%layers_(l)%m+0.1_dp*gw
        self%layers_(l)%v=0.999_dp*self%layers_(l)%v+0.001_dp*gw**2
        self%layers_(l)%mb=0.9_dp*self%layers_(l)%mb+0.1_dp*gb
        self%layers_(l)%vb=0.999_dp*self%layers_(l)%vb+0.001_dp*gb**2
        self%layers_(l)%weights=self%layers_(l)%weights-rate*self%layers_(l)%m/(sqrt(self%layers_(l)%v)+1.e-8_dp)
        self%layers_(l)%bias=self%layers_(l)%bias-rate*self%layers_(l)%mb/(sqrt(self%layers_(l)%vb)+1.e-8_dp)
      end do
      if(loss>best-self%tol) then
        stale=stale+1
      else
        stale=0
      end if
      best=min(best,loss)
      if(stale>=self%n_iter_no_change) then
        self%converged_=.true.; exit
      end if
    end do
    self%n_iter_=min(it,self%max_iter); self%loss_curve_=self%loss_curve_(:self%n_iter_)
    self%loss_=self%loss_curve_(self%n_iter_)
  end subroutine
  subroutine mlpc_fit(self,x,y)
    class(mlp_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), allocatable :: target(:,:)
    integer :: i
    call check_xy(x,size(y)); self%classes_=unique_labels(y)
    call require(size(self%classes_)>=2,'MLP classification needs at least two classes')
    allocate(target(size(y),size(self%classes_))); target=0
    do i=1,size(y)
      target(i,label_index(self%classes_,y(i)))=1
    end do
    call train(self,x,target,.true.)
  end subroutine
  function mlpc_proba(self,x) result(prob)
    class(mlp_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:)
    type(activation_buffer), allocatable :: buffers(:)
    call forward(self,x,.true.,buffers); prob=buffers(size(self%layers_))%values
  end function
  function mlpc_predict(self,x) result(y)
    class(mlp_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i
    real(dp), allocatable :: prob(:,:)
    prob=self%predict_proba(x)
    do i=1,size(y)
      y(i)=self%classes_(maxloc(prob(i,:),dim=1))
    end do
  end function
  real(dp) function mlpc_score(self,x,y) result(v)
    class(mlp_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    v=accuracy_score(y,self%predict(x))
  end function
  subroutine mlpr_fit(self,x,y)
    class(mlp_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    call train(self,x,reshape(y,[size(y),1]),.false.)
  end subroutine
  function mlpr_predict(self,x) result(y)
    class(mlp_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: y(size(x,1))
    type(activation_buffer), allocatable :: buffers(:)
    call forward(self,x,.false.,buffers); y=buffers(size(self%layers_))%values(:,1)
  end function
  real(dp) function mlpr_score(self,x,y) result(v)
    class(mlp_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    v=r2_score(y,self%predict(x))
  end function
end module
