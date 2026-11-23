! Stochastic and L-BFGS MLP contracts: sklearn 1.6.1 neural_network/{_multilayer_perceptron,
! _stochastic_optimizers,_base}.py (BSD-3-Clause); retained upstream notices.
module sklearn_neural_network
  use sklearn_rbm, only: bernoulli_rbm
  use lbfgsb_module, only: setulb
  use, intrinsic :: iso_fortran_env, only: int64
  use sklearn_legacy_rng, only: legacy_random_state
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
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
    integer :: max_iter=200,random_state=0,n_iter_=0,n_iter_no_change=10,batch_size=0
    integer :: n_features_in_=0,n_outputs_=0,n_layers_=0,max_fun=15000,n_function_evaluations_=0
    integer(int64) :: t_=0
    real(dp) :: learning_rate_init=0.001_dp,alpha=0.0001_dp,tol=1.e-4_dp,loss_=0,best_loss_=0
    real(dp) :: momentum=.9_dp,power_t=.5_dp,beta_1=.9_dp,beta_2=.999_dp,epsilon=1.e-8_dp
    character(16) :: solver="adam",learning_rate="constant",out_activation_="identity"
    logical :: shuffle=.true.,nesterovs_momentum=.true.,warm_start=.false.
    integer :: no_improvement_count_=0
    logical :: multilabel_=.false.,optimizer_ready_=.false.
    character(16) :: optimizer_kind_='',optimizer_schedule_='constant'
    real(dp) :: optimizer_rate_=0,optimizer_eta0_=0,optimizer_beta1_=0,optimizer_beta2_=0,optimizer_epsilon_=0
    real(dp) :: optimizer_momentum_=0,optimizer_power_=0
    logical :: optimizer_nesterov_=.false.
    integer(int64) :: optimizer_step_=0
    type(dense_layer), allocatable :: optimizer_saved_layers_(:)
    character(16) :: activation='relu'
    logical :: converged_=.false.
    type(dense_layer), allocatable :: layers_(:)
    real(dp), allocatable :: loss_curve_(:),gradient_(:),previous_parameters_(:)
    real(dp) :: previous_loss_=0
    character(60) :: optimization_message_=""
  end type
  type, extends(network_storage) :: mlp_classifier
    integer, allocatable :: classes_(:)
  contains
    procedure, private :: fit_vector=>mlpc_fit,fit_indicator=>mlpc_fit_indicator
    generic :: fit=>fit_vector,fit_indicator
    procedure, private :: partial_vector=>mlpc_partial_vector,partial_indicator=>mlpc_partial_indicator
    generic :: partial_fit=>partial_vector,partial_indicator
    procedure :: predict_proba=>mlpc_proba
    procedure :: predict_log_proba=>mlpc_log_proba
    procedure :: predict=>mlpc_predict
    procedure :: predict_indicator=>mlpc_predict_indicator
    procedure, private :: score_vector=>mlpc_score,score_indicator=>mlpc_score_indicator
    generic :: score=>score_vector,score_indicator
  end type
  type, extends(network_storage) :: mlp_regressor
  contains
    procedure, private :: fit_vector=>mlpr_fit,fit_matrix=>mlpr_fit_matrix
    generic :: fit=>fit_vector,fit_matrix
    procedure, private :: partial_vector=>mlpr_partial_vector,partial_matrix=>mlpr_partial_matrix
    generic :: partial_fit=>partial_vector,partial_matrix
    procedure :: predict=>mlpr_predict
    procedure :: predict_multioutput=>mlpr_predict_matrix
    procedure, private :: score_vector=>mlpr_score,score_matrix=>mlpr_score_matrix
    generic :: score=>score_vector,score_matrix
  end type
contains
  elemental real(dp) function mlp_logistic(x) result(v)
    real(dp), intent(in) :: x
    ! Match scipy.special.expit, including its negative overflow saturation,
    ! without evaluating an overflowing exponential under native FPE traps.
    if(x< -log(huge(1._dp))) then
      v=0
    else
      v=1/(1+exp(-x))
    end if
  end function
  function activate(x,kind) result(z)
    real(dp), intent(in) :: x(:,:)
    character(*), intent(in) :: kind
    real(dp) :: z(size(x,1),size(x,2))
    select case(trim(kind))
    case('relu'); z=max(0._dp,x)
    case('tanh'); z=tanh(x)
    case('logistic'); z=mlp_logistic(x)
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
  subroutine controls(self)
    class(network_storage), intent(inout) :: self
    call require(all(ieee_is_finite([self%learning_rate_init,self%alpha,self%tol,self%momentum, &
      self%power_t,self%beta_1,self%beta_2,self%epsilon])),'nonfinite MLP controls')
    call require(self%max_iter>0.and.self%learning_rate_init>0.and.self%alpha>=0.and.self%tol>=0, &
      'invalid MLP training controls')
    call require(self%max_fun>0,'MLP max_fun must be positive')
    call require(self%n_iter_no_change>0.and.self%batch_size>=0,'invalid MLP iteration or batch controls')
    call require(self%random_state>=0,'MLP random_state must be nonnegative')
    call require(self%momentum>=0.and.self%momentum<=1,'MLP momentum must be in [0,1]')
    call require(self%power_t>=0.and.self%beta_1>=0.and.self%beta_1<1.and. &
      self%beta_2>=0.and.self%beta_2<1.and.self%epsilon>0,'invalid MLP optimizer controls')
    call require(any(self%solver==['adam ','sgd  ','lbfgs']),'unknown MLP solver')
    call require(any(self%learning_rate==['constant  ','invscaling','adaptive  ']),'unknown MLP learning rate')
    call require(any(self%activation==['relu    ','tanh    ','logistic','identity']),'unknown MLP activation')
    if(.not.allocated(self%hidden_layer_sizes)) self%hidden_layer_sizes=[100]
    call require(all(self%hidden_layer_sizes>0),'hidden layer sizes must be positive')
  end subroutine
  subroutine initialize(self,p,outputs,rng)
    class(network_storage), intent(inout) :: self
    integer, intent(in) :: p,outputs
    integer, allocatable :: sizes(:)
    type(legacy_random_state), intent(inout) :: rng
    integer :: l,inputs,width,i,j
    real(dp) :: bound
    call controls(self)
    sizes=[p,self%hidden_layer_sizes,outputs]
    if(self%solver=='lbfgs'.and.self%optimizer_ready_.and.allocated(self%layers_)) then
      if(.not.allocated(self%optimizer_saved_layers_)) self%optimizer_saved_layers_=self%layers_
    end if
    if(allocated(self%layers_)) deallocate(self%layers_)
    allocate(self%layers_(size(sizes)-1)); call rng%seed(self%random_state)
    self%n_features_in_=p;self%n_outputs_=outputs;self%n_layers_=size(sizes);self%t_=0
    self%n_iter_=0;self%n_function_evaluations_=0;self%optimization_message_=''
    if(self%solver/='lbfgs') self%no_improvement_count_=0
    if(allocated(self%gradient_)) deallocate(self%gradient_)
    if(allocated(self%previous_parameters_)) deallocate(self%previous_parameters_)
    self%previous_loss_=0
    do l=1,size(self%layers_)
      inputs=sizes(l); width=sizes(l+1); bound=sqrt(6._dp/(inputs+width))
      if(self%activation=='logistic') bound=sqrt(2._dp/(inputs+width))
      allocate(self%layers_(l)%weights(inputs,width),self%layers_(l)%bias(width), &
        self%layers_(l)%m(inputs,width),self%layers_(l)%v(inputs,width), &
        self%layers_(l)%mb(width),self%layers_(l)%vb(width))
      ! NumPy draws row-major coefficients, then intercepts, one layer at a time.
      do i=1,inputs
        do j=1,width
          self%layers_(l)%weights(i,j)=-bound+2*bound*rng%uniform()
        end do
      end do
      do j=1,width
        self%layers_(l)%bias(j)=-bound+2*bound*rng%uniform()
      end do
      self%layers_(l)%m=0; self%layers_(l)%v=0; self%layers_(l)%mb=0; self%layers_(l)%vb=0
    end do
    if(self%solver=='lbfgs'.and.allocated(self%optimizer_saved_layers_)) call restore_optimizer(self,.false.)
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
      else if(classification.and.self%out_activation_=='logistic') then
        buffers(l)%values=mlp_logistic(buffers(l)%values)
      else if(classification) then
        do i=1,size(x,1)
          buffers(l)%values(i,:)=exp(buffers(l)%values(i,:)-logsumexp(buffers(l)%values(i,:)))
        end do
      end if
    end do
  end subroutine
  subroutine unpack_parameters(self,point)
    class(network_storage), intent(inout) :: self
    real(dp), intent(in) :: point(:)
    integer :: l,start,width,inputs,outputs
    start=1
    do l=1,size(self%layers_)
      inputs=size(self%layers_(l)%weights,1);outputs=size(self%layers_(l)%weights,2);width=inputs*outputs
      self%layers_(l)%weights=transpose(reshape(point(start:start+width-1),[outputs,inputs]))
      start=start+width
    end do
    do l=1,size(self%layers_)
      width=size(self%layers_(l)%bias);self%layers_(l)%bias=point(start:start+width-1);start=start+width
    end do
  end subroutine
  subroutine full_loss_gradient(self,x,target,classification,loss,gradient)
    class(network_storage), intent(in) :: self
    real(dp), intent(in) :: x(:,:),target(:,:)
    logical, intent(in) :: classification
    real(dp), intent(out) :: loss
    real(dp), allocatable, intent(out) :: gradient(:)
    type(activation_buffer), allocatable :: buffers(:)
    real(dp), allocatable :: prob(:,:),gw(:,:)
    real(dp) :: penalty,eps
    integer :: l,last,n,d,start,width
    n=size(x,1);last=size(self%layers_);penalty=0;d=0
    call forward(self,x,classification,buffers)
    do l=1,last
      penalty=penalty+sum(self%layers_(l)%weights**2)
      d=d+size(self%layers_(l)%weights)+size(self%layers_(l)%bias)
    end do
    if(classification) then
      eps=epsilon(1._dp);prob=max(eps,min(1-eps,buffers(last)%values))
      loss=-sum(target*log(prob))/n
      if(self%out_activation_=='logistic') loss=loss-sum((1-target)*log(1-prob))/n
    else
      loss=sum((buffers(last)%values-target)**2)/(2*real(n,dp)*size(target,2))
    end if
    loss=loss+self%alpha*penalty/(2*n)
    buffers(last)%delta=buffers(last)%values-target
    do l=last-1,1,-1
      buffers(l)%delta=matmul(buffers(l+1)%delta,transpose(self%layers_(l+1)%weights))* &
        derivative(buffers(l)%values,self%activation)
    end do
    allocate(gradient(d));start=1
    do l=1,last
      gw=(matmul(transpose(buffers(l-1)%values),buffers(l)%delta)+self%alpha*self%layers_(l)%weights)/n
      width=size(gw);gradient(start:start+width-1)=reshape(transpose(gw),[width]);start=start+width
    end do
    do l=1,last
      width=size(self%layers_(l)%bias)
      gradient(start:start+width-1)=sum(buffers(l)%delta,dim=1)/n;start=start+width
    end do
    call require(ieee_is_finite(loss).and.all(ieee_is_finite(gradient)),'MLP objective overflow; scale the inputs')
  end subroutine
  subroutine train_lbfgs(self,x,target,classification,preserve_history)
    class(network_storage), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),target(:,:)
    logical, intent(in) :: classification,preserve_history
    integer, parameter :: memory=10
    real(dp), allocatable :: point(:),g(:),wa(:),lower(:),upper(:),accepted(:)
    integer, allocatable :: iwa(:),nbd(:)
    real(dp) :: f,dsave(29)
    integer :: isave(44),d,l,start,width
    integer(int64) :: dimension
    logical :: lsave(4)
    character(60) :: task,csave
    dimension=0
    do l=1,size(self%layers_)
      dimension=dimension+int(size(self%layers_(l)%weights),int64)+size(self%layers_(l)%bias)
    end do
    call require(dimension<=(huge(d)-11*memory*memory-8*memory)/(2*memory+5), &
      'MLP optimizer dimension exceeds integer capacity')
    d=int(dimension)
    allocate(point(d),g(d),wa((2*memory+5)*d+11*memory*memory+8*memory), &
      iwa(3*d),nbd(d),lower(d),upper(d))
    start=1
    do l=1,size(self%layers_)
      width=size(self%layers_(l)%weights)
      point(start:start+width-1)=reshape(transpose(self%layers_(l)%weights),[width]);start=start+width
    end do
    do l=1,size(self%layers_)
      width=size(self%layers_(l)%bias);point(start:start+width-1)=self%layers_(l)%bias;start=start+width
    end do
    nbd=0;lower=0;upper=0;wa=0;iwa=0;isave=0;dsave=0;lsave=.false.;csave='';task='START';f=0;g=0
    accepted=point;self%previous_parameters_=point
    self%converged_=.false.
    ! Source L-BFGS leaves any pre-existing stochastic history and optimizer intact.
    do
      ! SciPy minimize defaults: memory=10, factr=1e7, maxls=20; tol maps to gtol.
      call setulb(d,memory,point,lower,upper,nbd,f,g,1.e7_dp,self%tol,wa,iwa,task,-1,csave,lsave,isave,dsave)
      if(task(:2)=='FG') then
        call unpack_parameters(self,point)
        call full_loss_gradient(self,x,target,classification,f,g)
        self%n_function_evaluations_=self%n_function_evaluations_+1
      else if(task(:5)=='NEW_X') then
        self%previous_parameters_=accepted;self%previous_loss_=dsave(2);accepted=point
        self%n_iter_=self%n_iter_+1
        if(self%n_iter_>=self%max_iter) then
          task='STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT'
        else if(self%n_function_evaluations_>self%max_fun) then
          task='STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT'
        end if
      else
        exit
      end if
    end do
    call unpack_parameters(self,point)
    call full_loss_gradient(self,x,target,classification,self%loss_,self%gradient_)
    if(self%n_iter_==0) self%previous_loss_=self%loss_
    self%converged_=task(:4)=='CONV';self%optimization_message_=task
  end subroutine
  subroutine reset_optimizer(self)
    class(network_storage), intent(inout) :: self
    integer :: l
    if(allocated(self%optimizer_saved_layers_)) deallocate(self%optimizer_saved_layers_)
    do l=1,size(self%layers_)
      self%layers_(l)%m=spread(spread(0._dp,1,size(self%layers_(l)%weights,1)),2,size(self%layers_(l)%weights,2))
      self%layers_(l)%v=self%layers_(l)%m
      self%layers_(l)%mb=spread(0._dp,1,size(self%layers_(l)%bias));self%layers_(l)%vb=self%layers_(l)%mb
    end do
    self%optimizer_kind_=self%solver;self%optimizer_schedule_=self%learning_rate
    self%optimizer_eta0_=self%learning_rate_init;self%optimizer_rate_=self%learning_rate_init
    self%optimizer_beta1_=self%beta_1;self%optimizer_beta2_=self%beta_2;self%optimizer_epsilon_=self%epsilon
    self%optimizer_momentum_=self%momentum;self%optimizer_nesterov_=self%nesterovs_momentum
    self%optimizer_power_=self%power_t;self%optimizer_step_=0;self%optimizer_ready_=.true.
  end subroutine
  subroutine restore_optimizer(self,consume)
    class(network_storage), intent(inout) :: self
    logical, intent(in) :: consume
    integer :: l
    if(allocated(self%optimizer_saved_layers_)) then
      if(consume) call require(size(self%optimizer_saved_layers_)==size(self%layers_), &
        'MLP cached optimizer layer count mismatch')
      do l=1,min(size(self%optimizer_saved_layers_),size(self%layers_))
        self%layers_(l)%m=self%optimizer_saved_layers_(l)%m;self%layers_(l)%v=self%optimizer_saved_layers_(l)%v
        self%layers_(l)%mb=self%optimizer_saved_layers_(l)%mb;self%layers_(l)%vb=self%optimizer_saved_layers_(l)%vb
      end do
      if(consume) deallocate(self%optimizer_saved_layers_)
    end if
    if(consume) then
      do l=1,size(self%layers_)
        call require(all(shape(self%layers_(l)%m)==shape(self%layers_(l)%weights)).and. &
          size(self%layers_(l)%mb)==size(self%layers_(l)%bias),'MLP cached optimizer parameter shape mismatch')
      end do
    end if
  end subroutine
  subroutine train(self,x,target,classification,incremental)
    class(network_storage), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),target(:,:)
    logical, intent(in) :: classification
    logical, optional, intent(in) :: incremental
    type(activation_buffer), allocatable :: buffers(:)
    type(legacy_random_state) :: rng
    real(dp), allocatable :: gw(:,:),gb(:),prob(:,:),xb(:,:),yb(:,:)
    integer, allocatable :: order(:)
    real(dp) :: loss,rate,penalty,accumulated,eps
    integer :: it,l,last,n,stale,bs,first,finish,m,i,j,tmp,history_size,epochs
    integer(int64) :: step
    logical :: reuse,streaming
    call check_xy(x,size(target,1)); call check_x(target)
    streaming=.false.;if(present(incremental)) streaming=incremental
    n=size(x,1);reuse=allocated(self%layers_).and.(self%warm_start.or.streaming)
    epochs=self%max_iter;if(streaming) epochs=1
    if(reuse) then
      call controls(self)
      call require(size(x,2)==self%n_features_in_,'MLP warm fit feature count mismatch')
      call require(size(self%hidden_layer_sizes)==size(self%layers_)-1,'MLP warm fit layer count mismatch')
      if(self%solver=='lbfgs') then
        do l=1,size(self%hidden_layer_sizes)
          call require(self%hidden_layer_sizes(l)==size(self%layers_(l)%bias),'MLP warm L-BFGS layer width mismatch')
        end do
      end if
      call require(size(target,2)==size(self%layers_(size(self%layers_))%bias),'MLP warm fit output count mismatch')
      call rng%seed(self%random_state)
      self%n_iter_=0;self%n_function_evaluations_=0;self%optimization_message_=''
      if(allocated(self%gradient_)) deallocate(self%gradient_)
      if(allocated(self%previous_parameters_)) deallocate(self%previous_parameters_)
      self%previous_loss_=0
    else
      call initialize(self,size(x,2),size(target,2),rng)
    end if
    last=size(self%layers_)
    self%out_activation_='identity'
    if(classification) then
      self%out_activation_='softmax'
      if(size(target,2)==1.or.self%multilabel_) self%out_activation_='logistic'
    end if
    if(self%solver=='lbfgs') then
      call train_lbfgs(self,x,target,classification,reuse)
      return
    end if
    if(reuse) then
      call require(allocated(self%loss_curve_),'MLP stochastic history unavailable; use a cold fit after L-BFGS')
      history_size=size(self%loss_curve_)
      self%loss_curve_=[self%loss_curve_,spread(0._dp,1,epochs)]
    else
      history_size=0;self%loss_curve_=spread(0._dp,1,epochs);self%best_loss_=huge(1._dp)
    end if
    if(.not.streaming.or..not.self%optimizer_ready_) then
      call reset_optimizer(self)
    else
      call restore_optimizer(self,.true.)
    end if
    stale=self%no_improvement_count_;step=self%optimizer_step_
    self%converged_=.false.;rate=self%optimizer_rate_
    bs=min(200,n);if(self%batch_size>0) bs=min(self%batch_size,n)
    order=[(i,i=1,n)];eps=epsilon(1._dp)
    do it=1,epochs
      if(self%shuffle) then
        do i=n,2,-1
          j=rng%integer_index(i);tmp=order(i);order(i)=order(j);order(j)=tmp
        end do
      end if
      accumulated=0
      do first=1,n,bs
        finish=min(n,first+bs-1);m=finish-first+1
        xb=x(order(first:finish),:);yb=target(order(first:finish),:)
        call forward(self,xb,classification,buffers);penalty=0
        do l=1,last
          penalty=penalty+sum(self%layers_(l)%weights**2)
        end do
        if(classification) then
          prob=max(eps,min(1-eps,buffers(last)%values))
          loss=-sum(yb*log(prob))/m
          if(self%out_activation_=='logistic') loss=loss-sum((1-yb)*log(1-prob))/m
        else
          loss=sum((buffers(last)%values-yb)**2)/(2*real(m,dp)*size(target,2))
        end if
        loss=loss+self%alpha*penalty/(2*m);accumulated=accumulated+loss*m
        buffers(last)%delta=buffers(last)%values-yb
        do l=last-1,1,-1
          buffers(l)%delta=matmul(buffers(l+1)%delta,transpose(self%layers_(l+1)%weights))* &
            derivative(buffers(l)%values,self%activation)
        end do
        step=step+1
        if(self%optimizer_kind_=='adam') &
          rate=self%optimizer_eta0_*sqrt(1-self%optimizer_beta2_**step)/(1-self%optimizer_beta1_**step)
        do l=1,last
          gw=(matmul(transpose(buffers(l-1)%values),buffers(l)%delta)+self%alpha*self%layers_(l)%weights)/m
          gb=sum(buffers(l)%delta,dim=1)/m
          if(self%optimizer_kind_=='adam') then
            self%layers_(l)%m=self%optimizer_beta1_*self%layers_(l)%m+(1-self%optimizer_beta1_)*gw
            self%layers_(l)%v=self%optimizer_beta2_*self%layers_(l)%v+(1-self%optimizer_beta2_)*gw**2
            self%layers_(l)%mb=self%optimizer_beta1_*self%layers_(l)%mb+(1-self%optimizer_beta1_)*gb
            self%layers_(l)%vb=self%optimizer_beta2_*self%layers_(l)%vb+(1-self%optimizer_beta2_)*gb**2
            self%layers_(l)%weights=self%layers_(l)%weights- &
              rate*self%layers_(l)%m/(sqrt(self%layers_(l)%v)+self%optimizer_epsilon_)
            self%layers_(l)%bias=self%layers_(l)%bias-rate*self%layers_(l)%mb/(sqrt(self%layers_(l)%vb)+self%optimizer_epsilon_)
          else
            self%layers_(l)%m=self%optimizer_momentum_*self%layers_(l)%m-rate*gw
            self%layers_(l)%mb=self%optimizer_momentum_*self%layers_(l)%mb-rate*gb
            if(self%optimizer_nesterov_) then
              self%layers_(l)%weights=self%layers_(l)%weights+self%optimizer_momentum_*self%layers_(l)%m-rate*gw
              self%layers_(l)%bias=self%layers_(l)%bias+self%optimizer_momentum_*self%layers_(l)%mb-rate*gb
            else
              self%layers_(l)%weights=self%layers_(l)%weights+self%layers_(l)%m
              self%layers_(l)%bias=self%layers_(l)%bias+self%layers_(l)%mb
            end if
          end if
          call require(all(ieee_is_finite(self%layers_(l)%weights)).and. &
            all(ieee_is_finite(self%layers_(l)%bias)),'MLP fit overflow; scale the inputs')
        end do
      end do
      self%loss_=accumulated/n;self%loss_curve_(history_size+it)=self%loss_;self%t_=self%t_+n
      if(self%loss_>self%best_loss_-self%tol) then
        stale=stale+1
      else
        stale=0
      end if
      self%best_loss_=min(self%best_loss_,self%loss_)
      if(self%optimizer_kind_=='sgd'.and.self%optimizer_schedule_=='invscaling') &
        rate=self%optimizer_eta0_/(real(self%t_,dp)+1)**self%optimizer_power_
      if(stale>self%n_iter_no_change) then
        if(self%optimizer_kind_=='sgd'.and.self%optimizer_schedule_=='adaptive'.and.rate>1.e-6_dp) then
          rate=rate/5;stale=0
        else
          self%converged_=.true.;exit
        end if
      end if
    end do
    self%n_iter_=min(it,epochs);self%loss_curve_=self%loss_curve_(:history_size+self%n_iter_)
    self%no_improvement_count_=stale
    self%optimizer_rate_=rate;self%optimizer_step_=step
  end subroutine
  subroutine mlpc_fit(self,x,y)
    class(mlp_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), allocatable :: target(:,:)
    integer :: i
    integer, allocatable :: current_classes(:)
    call check_xy(x,size(y))
    if(self%warm_start.and.allocated(self%layers_)) &
      call require(.not.self%multilabel_,'MLP warm fit target kind mismatch')
    self%multilabel_=.false.;current_classes=unique_labels(y)
    if(self%warm_start.and.allocated(self%classes_)) then
      call require(size(current_classes)==size(self%classes_),'MLP warm fit class set mismatch')
      call require(all(current_classes==self%classes_),'MLP warm fit class set mismatch')
    else
      self%classes_=current_classes
    end if
    call require(size(self%classes_)>=2,'MLP classification needs at least two classes')
    if(size(self%classes_)==2) then
      allocate(target(size(y),1));target(:,1)=real(merge(1,0,y==self%classes_(2)),dp)
    else
      allocate(target(size(y),size(self%classes_))); target=0
      do i=1,size(y)
        target(i,label_index(self%classes_,y(i)))=1
      end do
    end if
    call train(self,x,target,.true.)
  end subroutine
  subroutine mlpc_fit_indicator(self,x,y)
    class(mlp_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:,:)
    integer :: j
    call check_xy(x,size(y,1));call require(size(y,2)>0,'MLP indicator target width must be positive')
    if(size(y,2)==1) then
      call mlpc_fit(self,x,y(:,1));return
    end if
    call require(all(y==0.or.y==1),'MLP indicator targets must be zero or one')
    if(self%warm_start.and.allocated(self%layers_)) then
      call require(self%multilabel_,'MLP warm fit target kind mismatch')
      call require(size(y,2)==self%n_outputs_,'MLP warm fit indicator width mismatch')
      call require(size(self%classes_)==size(y,2),'MLP warm fit class set mismatch')
      call require(all(self%classes_==[(j-1,j=1,size(y,2))]),'MLP warm fit class set mismatch')
    end if
    self%multilabel_=.true.;self%classes_=[(j-1,j=1,size(y,2))]
    call train(self,x,real(y,dp),.true.)
  end subroutine
  function mlpc_predict_indicator(self,x) result(y)
    class(mlp_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: y(:,:)
    real(dp), allocatable :: probability(:,:)
    call require(self%multilabel_,'MLP indicator prediction requires a multilabel model')
    probability=self%predict_proba(x);y=merge(1,0,probability>.5_dp)
  end function
  real(dp) function mlpc_score_indicator(self,x,y) result(v)
    class(mlp_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:,:)
    integer, allocatable :: prediction(:,:)
    call check_xy(x,size(y,1));call require(size(y,2)>0,'MLP indicator target width must be positive')
    if(size(y,2)==1.and..not.self%multilabel_) then
      v=mlpc_score(self,x,y(:,1));return
    end if
    call require(all(y==0.or.y==1),'MLP indicator targets must be zero or one')
    prediction=self%predict_indicator(x)
    call require(size(y,2)==size(prediction,2),'MLP indicator score width mismatch')
    v=real(count(all(y==prediction,dim=2)),dp)/size(y,1)
  end function
  function mlpc_proba(self,x) result(prob)
    class(mlp_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:)
    type(activation_buffer), allocatable :: buffers(:)
    call forward(self,x,.true.,buffers)
    if(self%n_outputs_==1) then
      allocate(prob(size(x,1),2));prob(:,2)=buffers(size(self%layers_))%values(:,1);prob(:,1)=1-prob(:,2)
    else
      prob=buffers(size(self%layers_))%values
    end if
  end function
  function mlpc_log_proba(self,x) result(prob)
    class(mlp_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:)
    ! sklearn returns -Inf for exactly zero probabilities.
    prob=self%predict_proba(x)
    block
      use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_negative_inf
      integer :: i,j
      do j=1,size(prob,2)
        do i=1,size(prob,1)
          if(prob(i,j)==0) then
            prob(i,j)=ieee_value(0._dp,ieee_negative_inf)
          else
            prob(i,j)=log(prob(i,j))
          end if
        end do
      end do
    end block
  end function
  function mlpc_predict(self,x) result(y)
    class(mlp_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i
    real(dp), allocatable :: prob(:,:)
    call require(.not.self%multilabel_,'use predict_indicator for multilabel MLP outputs')
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
    self%multilabel_=.false.
    call train(self,x,reshape(y,[size(y),1]),.false.)
  end subroutine
  function mlpr_predict(self,x) result(y)
    class(mlp_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: y(size(x,1))
    type(activation_buffer), allocatable :: buffers(:)
    call forward(self,x,.false.,buffers)
    call require(self%n_outputs_==1,'use predict_multioutput for multiple MLP responses')
    y=buffers(size(self%layers_))%values(:,1)
  end function
  real(dp) function mlpr_score(self,x,y) result(v)
    class(mlp_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    v=mlpr_score_matrix(self,x,reshape(y,[size(y),1]))
  end function
  subroutine mlpr_fit_matrix(self,x,y)
    class(mlp_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    self%multilabel_=.false.;call train(self,x,y,.false.)
  end subroutine
  function mlpr_predict_matrix(self,x) result(y)
    class(mlp_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:)
    type(activation_buffer), allocatable :: buffers(:)
    call forward(self,x,.false.,buffers);y=buffers(size(self%layers_))%values
  end function
  real(dp) function mlpr_score_matrix(self,x,y) result(v)
    use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
    class(mlp_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), allocatable :: prediction(:,:)
    integer :: j
    call check_xy(x,size(y,1));call check_x(y)
    prediction=self%predict_multioutput(x)
    call require(size(y,2)==size(prediction,2),'MLP response score width mismatch')
    if(size(y,1)<2) then
      v=ieee_value(0._dp,ieee_quiet_nan);return
    end if
    v=0
    do j=1,size(y,2)
      v=v+r2_score(y(:,j),prediction(:,j))
    end do
    v=v/size(y,2)
  end function

  subroutine partial_classes(self,classes)
    class(mlp_classifier), intent(inout) :: self
    integer, optional, intent(in) :: classes(:)
    integer, allocatable :: declared(:)
    call require(self%solver/='lbfgs','MLP classifier partial_fit requires a stochastic solver')
    call controls(self)
    if(present(classes)) then
      call require(size(classes)>0,'MLP partial_fit classes must be nonempty')
      declared=unique_labels(classes)
      if(allocated(self%classes_)) then
        call require(size(declared)==size(self%classes_),'MLP partial_fit class declaration mismatch')
        call require(all(declared==self%classes_),'MLP partial_fit class declaration mismatch')
      else
        self%classes_=declared
      end if
    else
      call require(allocated(self%classes_),'MLP first partial_fit requires classes')
    end if
    call require(size(self%classes_)>=2,'MLP partial_fit needs at least two declared classes')
  end subroutine
  subroutine mlpc_partial_vector(self,x,y,classes)
    class(mlp_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    integer, optional, intent(in) :: classes(:)
    real(dp), allocatable :: target(:,:)
    integer, allocatable :: current(:)
    integer :: i,j
    call partial_classes(self,classes);call check_xy(x,size(y))
    if(allocated(self%layers_)) call require(.not.self%multilabel_,'MLP partial_fit target kind mismatch')
    current=unique_labels(y)
    if(self%warm_start) then
      call require(size(current)==size(self%classes_),'MLP warm fit class set mismatch')
      call require(all(current==self%classes_),'MLP warm fit class set mismatch')
    end if
    do i=1,size(current)
      call require(any(self%classes_==current(i)),'MLP partial_fit contains an undeclared class')
    end do
    self%multilabel_=.false.
    if(size(self%classes_)==2) then
      allocate(target(size(y),1));target(:,1)=real(merge(1,0,y==self%classes_(2)),dp)
    else
      allocate(target(size(y),size(self%classes_)));target=0
      do i=1,size(y)
        j=label_index(self%classes_,y(i));target(i,j)=1
      end do
    end if
    call train(self,x,target,.true.,.true.)
  end subroutine
  subroutine mlpc_partial_indicator(self,x,y,classes)
    class(mlp_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:,:)
    integer, optional, intent(in) :: classes(:)
    integer :: j
    if(size(y,2)==1) then
      call mlpc_partial_vector(self,x,y(:,1),classes);return
    end if
    call partial_classes(self,classes);call check_xy(x,size(y,1))
    call require(size(y,2)>0,'MLP indicator target width must be positive')
    call require(all(y==0.or.y==1),'MLP indicator targets must be zero or one')
    if(allocated(self%layers_)) then
      call require(self%multilabel_,'MLP partial_fit target kind mismatch')
      call require(size(y,2)==self%n_outputs_,'MLP warm fit indicator width mismatch')
    end if
    if(self%warm_start) call require(size(self%classes_)==size(y,2),'MLP warm fit class set mismatch')
    do j=0,size(y,2)-1
      call require(any(self%classes_==j),'MLP partial_fit contains an undeclared class')
    end do
    self%multilabel_=.true.;call train(self,x,real(y,dp),.true.,.true.)
  end subroutine
  subroutine mlpr_partial_vector(self,x,y)
    class(mlp_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    self%multilabel_=.false.;call train(self,x,reshape(y,[size(y),1]),.false.,.true.)
  end subroutine
  subroutine mlpr_partial_matrix(self,x,y)
    class(mlp_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    self%multilabel_=.false.;call train(self,x,y,.false.,.true.)
  end subroutine

end module
