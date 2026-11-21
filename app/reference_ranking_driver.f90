program reference_ranking_driver
  use sklearn, only: dp,roc_curve,precision_recall_curve,det_curve,auc,roc_auc_score,average_precision_score, &
    brier_score_loss,calibration_curve
  implicit none
  integer, allocatable :: labels(:)
  real(dp), allocatable :: scores(:),prob(:),weights(:),ax(:),ay(:)
  real(dp), allocatable :: a(:),b(:),c(:)
  integer :: n,m,pos,bins,strategy_code,i,unit,out
  real(dp) :: limit,partial_value
  logical :: explicit,weighted,drop_roc,drop_pr,do_det,do_scores,do_prob,do_partial
  character(16) :: mode
  character(8) :: strategy
  character(1024) :: input,output
  call get_command_argument(1,input); call get_command_argument(2,output); call get_command_argument(3,mode)
  open(newunit=unit,file=trim(input),status='old')
  read(unit,*) n,m,pos,bins,strategy_code,limit
  read(unit,*) explicit,weighted,drop_roc,drop_pr,do_det,do_scores,do_prob,do_partial
  allocate(labels(n),scores(n),prob(n),weights(n),ax(m),ay(m))
  read(unit,*) labels; read(unit,*) scores; read(unit,*) prob; read(unit,*) weights
  read(unit,*) ax; read(unit,*) ay; close(unit)
  strategy='uniform'; if(strategy_code==1) strategy='quantile'
  open(newunit=out,file=trim(output),status='replace')
  if(explicit.and.weighted) then
    call evaluate(pos,weights)
  else if(explicit) then
    call evaluate(pos)
  else if(weighted) then
    call evaluate(sample_weight=weights)
  else
    call evaluate()
  end if
  write(out,'(a,*(es26.17e3,1x))') 'generic_auc ',auc(ax,ay)
  write(out,'(a,*(es26.17e3,1x))') 'preserved ',real(labels,dp),scores,prob,weights,ax,ay
  close(out)
contains
  subroutine evaluate(pos_label,sample_weight)
    integer, intent(in), optional :: pos_label
    real(dp), intent(in), optional :: sample_weight(:)
    if(mode=='partial') then
      partial_value=roc_auc_score(labels,scores,pos_label,sample_weight,limit)
      error stop 'partial AUC unexpectedly accepted'
    end if
    call roc_curve(labels,scores,a,b,c,pos_label,sample_weight,drop_roc)
    call emit('roc_fpr',a); call emit('roc_tpr',b); call emit('roc_thresholds',c)
    call precision_recall_curve(labels,scores,a,b,c,pos_label,sample_weight,drop_pr)
    call emit('pr_precision',a); call emit('pr_recall',b); call emit('pr_thresholds',c)
    if(do_det) then
      call det_curve(labels,scores,a,b,c,pos_label,sample_weight)
      call emit('det_fpr',a); call emit('det_fnr',b); call emit('det_thresholds',c)
    end if
    if(do_scores) then
      call emit('roc_auc',[roc_auc_score(labels,scores,pos_label,sample_weight)])
      if(do_partial) call emit('partial_auc',[roc_auc_score(labels,scores,pos_label,sample_weight,limit)])
    end if
    if(do_scores.or.do_prob) &
      call emit('average_precision',[average_precision_score(labels,scores,pos_label,sample_weight)])
    if(do_prob) then
      call emit('brier',[brier_score_loss(labels,prob,pos_label,sample_weight)])
      call calibration_curve(labels,prob,a,b,bins,trim(strategy),pos_label)
      call emit('calibration_true',a); call emit('calibration_pred',b)
    end if
  end subroutine
  subroutine emit(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:)
    write(out,'(a,1x,*(es26.17e3,1x))') name,values
  end subroutine
end program
